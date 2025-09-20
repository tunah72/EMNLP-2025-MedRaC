from typing import List, Tuple, Optional, Dict
from pathlib import Path
import os
import json
import logging
import pandas as pd
import numpy as np
from method.method import Method
from model.model import LLM
from evaluator import Evaluator
from schema.schemas import prompt_style_to_schema
from openai import AsyncOpenAI
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv


EMBEDDINGS_PATH = "dataset_cleaned/embedding_key_data_new.jsonl"

logger = logging.getLogger(__name__)

class MedPrompt(Method):
    """
    kNN-based few-shot or direct medical calculation prompting.
    Implements generate_raw and evaluate per Method API using multiple Evaluators.
    """

    def __init__(
        self,
        prompt_style: str,
        llms: List[LLM],
        evaluators: List[Evaluator],
        num_examples: int,
        examples_file_path: str = EMBEDDINGS_PATH,
        embeddings_rate_limit: int = 10000,
        **kwargs,
    ) -> None:
        assert prompt_style in ['direct', 'cot', 'modular', 'modular_cot'], f"Unsupported prompt style: {prompt_style}"
        self.prompt_style = prompt_style
        self.evaluators = evaluators
        self.example_fp = examples_file_path
        self.num_examples = num_examples
        self.embeddings_rate_limit = embeddings_rate_limit

        # Tracking
        self.responses: Dict[str, List[str]] = {}
        self.input_tokens: Dict[str, List[int]] = {}
        self.output_tokens: Dict[str, List[int]] = {}
        self.prompts: List[Tuple[str, str]] = []

        # Store latest run artefacts (optional, mainly for  backward compatibility)
        self.answers: Dict[str, List[str]] = {}
        self.correctness: Dict[str, Dict[str, List[bool]]] = {}

        super().__init__(llms=llms, **kwargs)

    def generate_raw(
        self,
        test: bool = False,
        raw_json_dir: str = "raw_output",
    ) -> str:
        # Load data
        self.df = self.load_data_test() if test else self.load_dataset()

        # Build prompts
        notes = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        self.prompts = self._gen_prompt(notes, questions)

        # Ensure output dir
        Path(raw_json_dir).mkdir(parents=True, exist_ok=True)

        model_to_path: Dict[str, str] = {}
        for llm in self.llm_list:
            model_name = getattr(llm, 'model_name_full', llm.get_model_name())
            safe_name = model_name.replace('/', '_')
            schema = prompt_style_to_schema(self.prompt_style)
            generations = llm.generate(
                self.prompts,
                schema=schema,
                batch_size=self.batch_size,
            )

            resp, inp, outp = map(list, zip(*generations))
            self.responses[model_name]    = resp
            self.input_tokens[model_name] = inp
            self.output_tokens[model_name]= outp

            raw_records = self._build_records(model_name=model_name, include_evaluation=False)
            raw_path = Path(raw_json_dir) / f"{safe_name}_{self.prompt_style}_raw.json"
            self._dump_json(raw_records, raw_path)
            logger.info("Raw generation complete – %s (%d rows)", raw_path, len(raw_records))
            model_to_path[model_name] = str(raw_path)

        return (next(iter(model_to_path.values()))
                if len(model_to_path) == 1
                else json.dumps(model_to_path, indent=2))

    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output",
    ) -> str:
        raw_path = Path(raw_json_file)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        with raw_path.open('r', encoding='utf-8') as f:
            records = json.load(f)
        self.df = pd.DataFrame(records)

        model_name = self.df["Model Name"].iloc[0]
        self.responses[model_name]    = self.df["LLM Original Answer"].tolist()
        self.input_tokens[model_name] = self.df["Input Tokens"].tolist()
        self.output_tokens[model_name]= self.df["Output Tokens"].tolist()
        self.correctness[model_name] = {}

        for evaluator in self.evaluators:
            key, results = evaluator.check_correctness(
                responses=self.responses[model_name],
                ground_truths=self.df["Ground Truth Answer"].tolist(),
                calids=self.df["Calculator ID"].astype(str).tolist(),
                upper_limits=self.df["Upper Limit"].tolist(),
                lower_limits=self.df["Lower Limit"].tolist(),
                relevant_entities=self.df.get("Relevant Entities", []).tolist(),
                ground_truth_explanations=self.df.get("Ground Truth Explanation", []).tolist(),
            )
            self.correctness[model_name][key] = results
            self.df[key] = results

        Path(eval_json_dir).mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace('/', '_')
        eval_path = Path(eval_json_dir) / f"{safe_name}_{self.prompt_style}_eval.json"
        full_records = self._build_records(model_name=model_name, include_evaluation=True)
        self._dump_json(full_records, eval_path)
        logger.info("Evaluation written to %s", eval_path)
        return str(eval_path)

    

    def _gen_prompt(self, notes: List[str], questions: List[str]) -> List[Tuple[str, str]]:
        assert len(notes) == len(questions)
        load_dotenv()
        client = AsyncOpenAI()

        interval = 60.0 / self.embeddings_rate_limit

        async def fetch_embedding(text: str) -> np.ndarray:
            resp = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
            )
            # enforce spacing between calls
            await asyncio.sleep(interval)
            return np.array(resp.data[0].embedding)

        async def fetch_all_embeddings(texts: List[str]) -> List[np.ndarray]:
            embeddings = []
            for t in tqdm(texts, desc="Fetching embeddings"):
                embeddings.append(await fetch_embedding(t))
            return embeddings

        # prepare the queries
        queries = [f"Note: {n.strip()}\nQuestion: {q.strip()}"
                for n, q in zip(notes, questions)]
        # run them with rate-limit
        query_embeddings = asyncio.run(fetch_all_embeddings(queries))

        # now your existing kNN logic, e.g.:
        examples = []
        with open(self.example_fp, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                examples.append({
                    'embedding': np.array(obj['embedding']),
                    'value': obj['value'],
                })
        emb_matrix = np.stack([ex["embedding"] for ex in examples])
        # normalize once:
        emb_norms = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)

        prompts = []
        for note, question, q_emb in zip(notes, questions, query_embeddings):
            # cosine sims & top_k as before
            q_norm = q_emb / np.linalg.norm(q_emb)
            sims = emb_norms @ q_norm
            top_k = sorted(zip(sims, examples), key=lambda x: -x[0])[: self.num_examples]

            # build example prefix
            example_prefix = ""
            for _, rec in top_k:
                val = rec['value']
                example_prefix += (
                    f"Given the following patient note:\n{val['patient_note']}\n"
                    f"We want to find {val['question']}\n"
                    f"extract: {val['relevant_entities']}\n"
                    f"reasoning: {val['ground_truth_explanation']}\n"
                    f"answer: {val['ground_truth_answer']}\n\n"
                )
            example_prefix += f"Now for the new case:\nNote: {note}\nQuestion: {question}\n\n"

            # dispatch to your static prompt builder
            # e.g. self.direct expects lists, so adapt to single-item
            base = getattr(self, self.prompt_style)(
                [""], [note], [question]
            )[0]  # returns (system_msg, user_msg)
            system_msg, user_msg = base
            prompts.append((system_msg, example_prefix + user_msg))

        return prompts

