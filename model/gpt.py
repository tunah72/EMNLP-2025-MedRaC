import os
import json
import asyncio
import time
import tiktoken
from typing import List, Tuple, Optional, Type
from collections import deque
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI as DeepseekClient, LengthFinishReasonError
from transformers import AutoTokenizer
from model.model import LLM
import concurrent.futures
from tqdm import tqdm
import nest_asyncio
nest_asyncio.apply()

class APIModel(LLM):
    def __init__(
        self,
        model_name: str,
        temperature: float = 1.0,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
    ):
        """
        Args:
            model_name: GPT model name, e.g., "OpenAI/gpt-4o-mini" or "deepseek-chat"
            temperature: sampling temperature
            rpm_limit: max requests per rolling 60-second window
            tpm_limit: max tokens (in+out) per rolling 60-second window
        """
        super().__init__(model_name=model_name, temperature=temperature)
        load_dotenv()

        # rate-limiting state
        self._req_times   = deque()               # timestamps of recent requests
        self._token_times = deque()               # (timestamp, tokens_used) entries
        self._throttle_lock = asyncio.Lock()

        max_tokens_map = {
            "gpt-4o-mini": 16384,
            "gpt-4.1-mini": 32768,
            "gpt-4o":      16384,
            "gpt-o3-mini": 100000,
            "deepseek-reasoner": 8000,
            "deepseek-chat": 8000,
        }

        # rate-limiting defaults (Open AI Tier 1)
        rpm_map = {
            "gpt-4o-mini": 500,
            "gpt-4o":      500,
            "gpt-o3-mini": 500,
            "gpt-o4-mini": 500,
        }

        tpm_map = {
            "gpt-4o-mini": 200000,
            "gpt-4o":      200000,
            "gpt-o3-mini": 200000,
            "gpt-o4-mini": 200000,
        }

        self.max_tokens = max_tokens_map.get(self.model_name, 16834)
        self.rpm_limit  = rpm_limit or rpm_map.get(self.model_name, 500)
        self.tpm_limit  = tpm_limit or tpm_map.get(self.model_name, 200000)
        self.tokens_used = 0
        self.requests_made = 0
        print(f'{self.model_name} max_tokens: {self.max_tokens}, rpm_limit: {self.rpm_limit}, tpm_limit: {self.tpm_limit}')


        if "deepseek" in self.model_name.lower():
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            self.deepseek_client  = DeepseekClient(
                api_key=self.deepseek_api_key,
                base_url="https://api.deepseek.com",
            )
            self.deepseek_executor = concurrent.futures.ThreadPoolExecutor(max_workers=100)
            self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
        else:
            self.api_key  = os.getenv("OPENAI_API_KEY")
            self.client   = AsyncOpenAI(api_key=self.api_key)
            # self.api_key = os.getenv("API_KEY")
            # self.client = AsyncOpenAI(
            #     base_url="https://pro.xiaoai.plus/v1",
            #     api_key=self.api_key,
            # )
            try:
                # primary tokenizer for this model
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # if tiktoken doesn’t have an encoding for this model name
                print(f"⚠️ No tiktoken encoding found for '{self.model_name}'. Falling back to 'cl100k_base'.")
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def _throttle(self, tokens_est: int):
        async with self._throttle_lock:
            now = time.monotonic()

            # purge old entries
            while self._req_times and now - self._req_times[0] > 60:
                self._req_times.popleft()
            while self._token_times and now - self._token_times[0][0] > 60:
                self._token_times.popleft()

            # calculate how long until we fall back under RPM
            rpm_wait = 0
            if self.rpm_limit and len(self._req_times) >= self.rpm_limit:
                oldest = self._req_times[0]
                rpm_wait = (oldest + 60) - now

            # throttle by TPM
            tpm_wait = 0
            if self.tpm_limit:
                used = sum(toks for _, toks in self._token_times)
                needed = used + tokens_est - self.tpm_limit
                if needed > 0:
                    # walk through the oldest entries until we've freed >= needed
                    freed = 0
                    for ts, toks in self._token_times:
                        freed += toks
                        if freed >= needed:
                            # once we've freed enough, wait until this bucket expires
                            tpm_wait = (ts + 60) - now
                            break
                    else:
                        # if even all buckets aren’t enough, wait until the last one expires
                        last_ts = self._token_times[-1][0]
                        tpm_wait = (last_ts + 60) - now
            
        # if either limit is blocking, sleep the max required
        wait = max(rpm_wait, tpm_wait, 0)
        if wait > 0:
            await asyncio.sleep(wait)
            return await self._throttle(tokens_est)

        async with self._throttle_lock:
            self._req_times.append(now)
            reserve = [now, tokens_est]
            self._token_times.append(reserve)
            return reserve

    async def _generate_single(
        self,
        system_msg: str,
        user_msg: str,
        schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[object, int, int]:
        # 1) Throttle
        est_tokens = (self.tokens_used / self.requests_made) * 1.5 if self.requests_made != 0 else self.max_tokens
        reserve_entry = await self._throttle(est_tokens)

        # 2) Build shared params
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ]
        common = {
            "model":       self.model_name,
            "messages":    messages,
            "temperature": self.temperature
        }

        # 3) Call the right endpoint
        model_key = self.model_name.lower()
        if "deepseek" in model_key:
            # build req per-call
            def blocking_call():
                req = common.copy()
                req["max_completion_tokens"] = self.max_tokens
                if "reasoner" not in model_key and schema is not None:
                    req["response_format"] = {"type": "json_object"}
                return self.deepseek_client.chat.completions.create(**req)

            loop = asyncio.get_running_loop()
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    resp = await loop.run_in_executor(self.deepseek_executor, blocking_call)
                    break  
                except json.JSONDecodeError as je:
                    raw = getattr(self.deepseek_client, "_last_response", None)
                    body = raw.text if raw else "<no raw>"
                    print(f"⚠️ Deepseek parse JSON failed (try {attempt}/{max_retries}): {je}")
                    print("Response body:", body)
                except Exception as e:
                    print(f"⚠️ Deepseek request failed (try {attempt}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
            else:
                raise RuntimeError("Deepseek fails in 3 times, exit ┭┮﹏┭┮")

        else:
            # OpenAI beta endpoint
            req = common.copy()
            req["max_completion_tokens"] = self.max_tokens
            if schema is not None:
                # Pydantic model class
                req["response_format"] = schema
            try:
                # first, try the schema‐enforced parse
                resp = await self.client.beta.chat.completions.parse(**req)
                # resp = await self.client.chat.completions.create(**req)

            except LengthFinishReasonError as e:
                print(f"⚠️ Schema parse failed due to length: {e}. Falling back to raw text.")
                # drop the schema directive
                req.pop("response_format", None)
                # optionally shrink the allowed tokens a bit
                req["max_completion_tokens"] = max(self.max_tokens - 50, 0)

                # do a raw completion
                raw_resp = await self.client.chat.completions.create(**req)

                # wrap it so downstream code (usage + choices) still works
                class _RawWrapper:
                    def __init__(self, raw):
                        self.usage   = raw.usage
                        self.choices = raw.choices
                resp = _RawWrapper(raw_resp)

            except Exception as e:
                # any other parse error (e.g. JSONDecodeError)
                print(f"⚠️ Schema parse failed ({type(e).__name__}): {e}. Falling back to raw text.")
                req.pop("response_format", None)
                raw_resp = await self.client.chat.completions.create(**req)
                resp = _RawWrapper(raw_resp)


        # 4) Extract tokens
        input_tokens  = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        async with self._throttle_lock:
            total = input_tokens + output_tokens
            self.tokens_used += total
            self.requests_made += 1
            reserve_entry[1] = total

        # 5) Normalize the payload
        choice = resp.choices[0].message
        if hasattr(choice, "parsed") and isinstance(choice.parsed, BaseModel):
            result = choice.parsed.model_dump()
        else:
            text = choice.content
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                result = text

        return result, input_tokens, output_tokens

    async def generate_async(
        self,
        prompts: List[Tuple[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True
    ) -> List[Tuple[str, int, int]]:

        # 1) a tiny wrapper to remember each prompt's index
        async def _wrap(idx: int, sys_msg: str, user_msg: str):
            out = await self._generate_single(sys_msg, user_msg, schema)
            return idx, out

        # 2) kick off one wrapped task per prompt
        tasks = [
            asyncio.create_task(_wrap(i, sys, usr))
            for i, (sys, usr) in enumerate(prompts)
        ]

        # 3a) fast path: no progress bar
        if not show_progress:
            done = await asyncio.gather(*tasks)
            # sort by idx and drop the index
            done.sort(key=lambda x: x[0])
            return [res for _, res in done]

        # 3b) progress‐bar path
        results: List[Optional[Tuple[str,int,int]]] = [None] * len(prompts)
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"⏳ Generating From {self.model_name}…"
        ):
            idx, res = await coro
            results[idx] = res

        return results  # fully populated, in-order


    def generate(
        self,
        prompts: List[Tuple[str, str]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[Tuple[str, int, int]]:
        # forward the show_progress flag
        return asyncio.run(
            self.generate_async(prompts, schema, show_progress=show_progress)
        )