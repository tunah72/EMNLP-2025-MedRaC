from __future__ import annotations

"""Self-Consistency evaluation module

This refactor splits the former ``evaluate`` routine into two distinct phases:

1. ``generate_raw`` – calls the LLM(s), captures raw responses together with
   input/output token counts, and writes **one JSONL file per voting round**.
   Only **generation-time artifacts** are recorded; no correctness judgment is
   made here.
2. ``evaluate`` – left blank for now, to be completed later with scoring /
   analysis logic that consumes the JSONL files produced by ``generate_raw``.

The JSONL schema of ``generate_raw`` follows ``Method._build_records`` exactly,
so every row in the original dataset remains aligned (positional indexing 0…N-1).
"""
from pathlib import Path
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from collections import Counter
import re, random, math

import numpy as np
from schema.schemas import prompt_style_to_schema

from method import Method
from model import LLM
from evaluator import RegEvaluator

__all__ = ["SelfConsistency"]

_DECIMAL_PATTERN = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*$")


class SelfConsistency(Method):
    """Run *self-consistency* prompting with an arbitrary number of LLMs."""

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    _VALID_PROMPT_STYLES = {
        "direct": "direct",
        "cot": "cot",
        "one_shot": "one_shot",
        "modular": "modular",
        "modular_cot": "modular_cot",
    }

    def __init__(
        self,
        prompt_style: str,
        llms: List[LLM],
        ans_parser: LLM,
        evaluator: RegEvaluator,
        voting_times: int = 1,
        **kwargs,
    ) -> None:
        if prompt_style not in self._VALID_PROMPT_STYLES:
            raise ValueError(f"Prompt style '{prompt_style}' not supported.")

        # Prompt construction callback (Method provides these helpers)
        self.prompt_style: str = prompt_style
        self.prompt_fn = getattr(self, self._VALID_PROMPT_STYLES[prompt_style])

        self.evaluator = evaluator
        self.voting_times = int(voting_times)
        self.llm_list = llms
        self.ans_parser = ans_parser

        # --- runtime containers ------------------------------------------------
        # keys are *model_name* **or** *model_name_voting<i>*
        self.responses: Dict[str, List[str]] = {}
        self.input_tokens: Dict[str, List[int]] = {}
        self.output_tokens: Dict[str, List[int]] = {}
        self.history: Dict[str, List[str]] | None = None  # optional external field

        # location of written JSONL files per model
        self.paths_by_model: Dict[str, List[str]] = defaultdict(list)

        super().__init__(llms=llms, **kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_raw(
        self,
        test: bool = False,
        output_dir: str | os.PathLike[str] = "raw_output/self_consistency",
    ) -> Dict[str, List[str]]:
        """Call each LLM ``voting_times`` times and dump raw generations.

        Parameters
        ----------
        test : bool, default=False
            If ``True`` the *test* split is used instead of the full training
            split (delegated to ``load_data_test``).
        output_dir : str | PathLike, default="json_output"
            Target directory. A fresh *JSONL* file is created for each voting
            round, e.g.::

                <output_dir>/gpt-4o_direct_voting1.jsonl
                <output_dir>/gpt-4o_direct_voting2.jsonl

        Returns
        -------
        dict
            Mapping *base model name* to the list of JSONL paths generated for
            that model.
        """
        # ------------------------------------------------------------------
        # 1)  Load data split
        # ------------------------------------------------------------------
        self.df = self.load_data_test() if test else self.load_dataset()

        notes = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()

        # ------------------------------------------------------------------
        # 2)  Build prompts once – identical for every LLM
        # ------------------------------------------------------------------
        prompts = self.prompt_fn(calids, notes, questions)

        # Ensure output directory exists
        output_dir += f"/{self.prompt_style}/"
        os.makedirs(output_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 3)  Iterate *model × voting* and collect generations
        # ------------------------------------------------------------------
        for llm in self.llm_list:
            model_name = llm.get_model_name()

            for vote_idx in range(1, self.voting_times + 1):
                key = f"{model_name}_voting{vote_idx}"

                # --- inference ------------------------------------------------
                gen_out = llm.generate(
                    prompts,
                    schema=prompt_style_to_schema(self.prompt_style),
                    batch_size=self.batch_size,
                )

                resp_seq, in_tok_seq, out_tok_seq = zip(*gen_out)  # transpose

                self.responses[key]      = list(resp_seq)
                self.input_tokens[key]   = list(in_tok_seq)
                self.output_tokens[key]  = list(out_tok_seq)

                # --- write JSONL --------------------------------------------
                records = self._build_records(model_name=key, include_evaluation=False)

                safe_model = model_name.replace("/", "_")
                out_path = os.path.join(
                    output_dir, f"{safe_model}_{self.prompt_style}_voting{vote_idx}.json"
                )

                self._dump_json(records, out_path)

                self.paths_by_model[model_name].append(out_path)
                print(f"[SelfConsistency] • wrote {len(records):,} rows → {out_path}")

        return self.paths_by_model

    def evaluate(
        self,
        raw_json_dir: str,
        eval_json_dir: str = "eval_output/self_consistency",
    ) -> str:
        """
        """
        os.makedirs(eval_json_dir, exist_ok=True)

        for fname in os.listdir(raw_json_dir):
            if not fname.endswith(".json"):
                continue

            raw_path  = Path(raw_json_dir) / fname
            eval_path = Path(eval_json_dir) / f"{raw_path.stem}_eval.json"

            with raw_path.open("r", encoding="utf-8") as f:
                records = json.load(f)

            # self.df = pd.DataFrame(records)

            raw_answers   = []
            ground_truths = []
            calids        = []
            upper_limits  = []
            lower_limits  = []

            for rec in records:
                resp_str = rec.get("LLM Original Answer", "")

                try:
                    obj = json.loads(resp_str)
                    raw_ans = obj["answer"] if isinstance(obj, dict) and "answer" in obj else resp_str
                except Exception:
                    raw_ans = resp_str

                raw_answers.append(str(raw_ans))
                ground_truths.append(rec["Ground Truth Answer"])
                calids.append(rec["Calculator ID"])
                upper_limits.append(rec["Upper Limit"])
                lower_limits.append(rec["Lower Limit"])

            # ---------- 2) parse answer using llm ----------
            parsing_prompts = self._parse_ans_prompt(raw_answers)
            gen_out = self.ans_parser.generate(
                parsing_prompts, batch_size=self.batch_size
            )

            resp_seq, in_tok_seq, out_tok_seq = zip(*gen_out)
            resp_seq = list(resp_seq)
            print(resp_seq)

            _, result_list = self.evaluator.check_correctness_parsed(
                responses     = resp_seq,
                ground_truths = ground_truths,
                calids        = calids,
                upper_limits  = upper_limits,
                lower_limits  = lower_limits,
            )

            for rec, p_ans, res in zip(records, resp_seq, result_list):
                rec["Parsed Answer"] = str(p_ans)
                rec["Result"]        = res

            with eval_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=4)

            print(f"[SelfConsistency] • evaluated {len(records):,} rows → {eval_path}")

        return str(eval_json_dir)

    # -------------------------------------------------------------
    # -- voting --
    # -------------------------------------------------------------
    def calculate_voting_accuracy(
            self,
            json_dir: str,                             # Directory storing predicted .json/.jsonl files
            output_dir: str = "stats/selfConsistency"  # Output directory
        ) -> None:

        # ---------- helpers ----------
        _DECIMAL_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")  # simple numeric text

        def _is_decimal(x) -> bool:
            """Return True if *x* looks like an int/float literal such as '12', '3.14' or -7.0."""
            return isinstance(x, (int, float)) or (
                isinstance(x, str) and bool(_DECIMAL_PATTERN.match(x.strip()))
            )

        def _to_float_safe(x) -> float:
            """Convert *x* to float, raising a clear error on failure."""
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                return float(x.strip())
            raise ValueError(f"Cannot convert to float: {x!r}")

        def _determine_step(ans_str: str) -> float:
            """
            Return interval width based on the number of decimal places **written**
            in *ans_str*.

            Examples
            --------
            '12'        -> 0 .5
            '12.3'      -> 0 .05
            '12.34'     -> 0 .005
            '12.3456'   -> 0 .00005
            """
            s = ans_str.strip()
            m = re.match(r"^[+-]?\d+(?:\.(\d+))?$", s)
            if m:
                dec_part = m.group(1) or ""
                prec = len(dec_part)
                return 0.5 * (10 ** -prec)
            # Fallback for scientific notation or unusual formats
            return 0.05  # default to one-decimal precision

        def _group_interval(
            val: float,
            low_lim: float,
            up_lim: float,
            step: float,
        ) -> Tuple[float, float]:
            """
            Assign *val* to an interval of size *step*, anchored at *low_lim* and
            clamped to [low_lim, up_lim].

            Parameters
            ----------
            val : float
                The numeric value to place into an interval.
            low_lim, up_lim : float
                Bounds for clamping *val* and for interval construction.
            step : float
                Interval width, e.g. 0.5 for integers, 0.05 for one-decimal numbers.
            """
            val = max(min(val, up_lim), low_lim)       # clamp value first
            n   = math.floor((val - low_lim) / step)
            lower = low_lim + n * step
            upper = min(lower + step, up_lim)

            # round to a sensible number of decimals (at least 4, or based on step)
            decs = max(4, int(abs(math.log10(step))) + 2)
            return round(lower, decs), round(upper, decs)

        def _label_interval(itv: Tuple[float, float]) -> str:
            """Return a human-readable label of *itv* such as '1.0-1.5'."""
            return f"{itv[0]}-{itv[1]}"

        # ---------- ① Load files ----------
        files = sorted(
            f for f in os.listdir(json_dir) if f.endswith((".json", ".jsonl"))
        )
        n_files = len(files)
        if n_files < 5:
            raise ValueError(f"Not enough files in directory (found only {n_files}): {json_dir}")
        print(f"[Voting] Detected {n_files} prediction files")

        def _load(fp: str) -> List[Dict]:
            with open(fp, "r", encoding="utf-8") as fh:
                txt = fh.read().strip()

            # try whole-file JSON
            try:
                data = json.loads(txt)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                pass

            # JSONL fallback
            recs = []
            for ln, line in enumerate(txt.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{fp} JSON parsing failed on line {ln}: {e}")
            return recs

        records_per_file = [_load(os.path.join(json_dir, f)) for f in files]

        # ---------- ② Validate consistent row counts ----------
        row_counts = {len(r) for r in records_per_file}
        if len(row_counts) != 1:
            raise ValueError(f"Inconsistent row counts across files: {row_counts}")
        n_rows = row_counts.pop()

        # ---------- ③ Common fields ----------
        ref = records_per_file[0]
        row_numbers   = [r["Row Number"]          for r in ref]
        calids        = [str(r["Calculator ID"])  for r in ref]
        categories    = [r["Category"]            for r in ref]
        lower_limits  = [r["Lower Limit"]         for r in ref]
        upper_limits  = [r["Upper Limit"]         for r in ref]
        ground_truths = [r["Ground Truth Answer"] for r in ref]

        # ---------- ④ Answers & tokens ----------
        answers_per_file, in_tok_per_file, out_tok_per_file = [], [], []
        for recs in records_per_file:
            ans, itok, otok = [], [], []
            for r in recs:
                ans.append(str(r.get("Parsed Answer", "")))
                itok.append(r.get("Input Tokens", 0))
                otok.append(r.get("Output Tokens", 0))
            answers_per_file.append(ans)
            in_tok_per_file.append(np.asarray(itok, dtype=float))
            out_tok_per_file.append(np.asarray(otok, dtype=float))

        # ---------- ⑤ Group label ----------
        def _group_label(row_idx: int, answer: str) -> str:
            low, up = lower_limits[row_idx], upper_limits[row_idx]
            if _is_decimal(answer):
                try:
                    val   = _to_float_safe(answer)
                    low_f = _to_float_safe(low)
                    up_f  = _to_float_safe(up)
                    step  = _determine_step(answer)
                    return _label_interval(_group_interval(val, low_f, up_f, step))
                except Exception:
                    pass
            return str(answer).strip()

        # ---------- ⑥ Output directory ----------
        os.makedirs(output_dir, exist_ok=True)

        # ---------- ⑦ Sampling voting ----------
        ks = [k for k in range(5, n_files + 1, 5)]
        for k in ks:
            comb_total = math.comb(n_files, k)
            n_samples  = min(100, comb_total)
            sampled_idx_sets = set()
            rng = random.Random(42 + k)

            acc_list, in_tok_avg, out_tok_avg = [], [], []

            # row-level stats
            vote_stats = [defaultdict(int) for _ in range(n_rows)]
            majority_overall = [""] * n_rows

            # Determine majority label across all files for each row
            for row in range(n_rows):
                labels = [_group_label(row, answers_per_file[i][row])
                        for i in range(n_files)]
                majority_overall[row] = Counter(labels).most_common(1)[0][0]

            while len(sampled_idx_sets) < n_samples:
                idxs = tuple(sorted(rng.sample(range(n_files), k)))
                if idxs in sampled_idx_sets:
                    continue
                sampled_idx_sets.add(idxs)

                majority_raw_ans = []
                for row in range(n_rows):
                    grp_cnt   = Counter()
                    raw_cnt_g = defaultdict(Counter)

                    for fi in idxs:
                        raw_ans = answers_per_file[fi][row]
                        g = _group_label(row, raw_ans)
                        grp_cnt[g] += 1
                        raw_cnt_g[g][raw_ans] += 1

                    # accumulate votes
                    for g, c in grp_cnt.items():
                        vote_stats[row][g] += c

                    top_group = grp_cnt.most_common(1)[0][0]
                    rep_ans   = raw_cnt_g[top_group].most_common(1)[0][0]
                    majority_raw_ans.append(rep_ans)

                # evaluate correctness via evaluator
                _, flags = self.evaluator.check_correctness_parsed(
                    majority_raw_ans,
                    ground_truths,
                    calids,
                    upper_limits,
                    lower_limits,
                )
                flags_num = [1 if str(f).lower().startswith("correct") else 0
                            for f in flags]
                acc_list.append(np.mean(flags_num))

                # token stats
                sel_in  = [in_tok_per_file[i]  for i in idxs]
                sel_out = [out_tok_per_file[i] for i in idxs]
                in_tok_avg.append(np.mean(np.concatenate(sel_in)))
                out_tok_avg.append(np.mean(np.concatenate(sel_out)))

            avg_acc = np.mean(acc_list) * 100
            print(f"[Voting k={k}] Selected {n_samples} combinations, average accuracy: {avg_acc:.2f}%")

            # ---------- ⑧ Main statistics ----------
            result_main = {
                "num_files_total"      : n_files,
                "voting_k"             : k,
                "sampled_combinations" : n_samples,
                "accuracy_average(%)"  : round(avg_acc, 2),
                "accuracy_highest(%)"  : round(np.max(acc_list) * 100, 2),
                "accuracy_lowest(%)"   : round(np.min(acc_list) * 100, 2),
                "input_tokens_average" : int(round(np.mean(in_tok_avg))),
                "output_tokens_average": int(round(np.mean(out_tok_avg))),
            }
            with open(os.path.join(output_dir, f"voting_{k}_of_{n_files}.json"),
                    "w", encoding="utf-8") as fh:
                json.dump(result_main, fh, indent=4, ensure_ascii=False)

            # ---------- ⑨ Detailed statistics ----------
            details = []
            for row, rn in enumerate(row_numbers):
                stats = {
                    g: round(c / n_samples, 2)          # average votes per sample
                    for g, c in sorted(vote_stats[row].items(),
                                    key=lambda x: (-x[1], x[0]))
                }
                details.append({
                    "Row Number"      : rn,
                    "Calculator ID"   : calids[row],
                    "Category"        : categories[row],
                    "Ground Truth"    : ground_truths[row],
                    "Answer Statistics(avg_votes)" : stats,
                    "Majority Answer / Interval"   : majority_overall[row],
                    "Total Votes per Sample"       : k,
                    "Samples Aggregated"           : n_samples,
                })

            with open(os.path.join(output_dir,
                                f"voting_{k}_of_{n_files}_details.json"),
                    "w", encoding="utf-8") as fh:
                json.dump(details, fh, indent=4, ensure_ascii=False)

            print(f"[Voting k={k}] Statistics file written to {output_dir} ✓")


    @staticmethod
    def _parse_ans_prompt(ans: List[str]) -> List[Tuple[str, str]]:
        """
        Create zero-shot chain-of-thought (CoT) prompts for multiple medical calculation tasks.

        Args:
            calids: List of IDs of the specific calculators.
            notes: List of patient note texts.
            questions: List of calculation questions.

        Returns:
            List of tuples, where each tuple is (system_message, user_message).
        """
        prompts = []

        for an in ans:
            system_msg = (
                "You are a helpful assistant for extracting numeric or date information from medical calculator outputs."
            )
            user_msg = (
            f"""
            The given text is the output of a medical calculator. Please ONLY extract numeric or date information, as follows:
            - If the answer involves numeric calculation, identify the number representing the final result, and return only that number (without units or extra content).
            - If the answer involves a date, identify the final date and return it only, in the format MM/DD/YYYY (e.g., 08/31/2023 or 07/03/2000), with no other text.
            - If the answer describes age or duration (in the form of weeks and days), return a tuple clearly stating the weeks and days, e.g., (4 weeks, 3 days), (0 weeks, 5 days).
            \n Given text: {an}
            """
            )
            prompts.append((system_msg, user_msg))

        return prompts
