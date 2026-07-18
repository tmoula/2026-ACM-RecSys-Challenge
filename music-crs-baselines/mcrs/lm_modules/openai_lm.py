"""OpenAI GPT response generation for conversational music recommendations."""

from __future__ import annotations

import os
import time
from typing import List, Optional

from openai import OpenAI

from .fewshot_pool import FewShotPool


class OpenAILM:
    """Generate grounded curator responses via OpenAI chat completions."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.75,
        top_p: float = 0.92,
        frequency_penalty: float = 0.45,
        presence_penalty: float = 0.35,
        fewshot_pool_path: str = "./cache/generation/fewshot_pool.json",
        num_fewshot: int = 2,
        max_retries: int = 4,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.num_fewshot = num_fewshot
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.fewshot_pool = FewShotPool(fewshot_pool_path)

    def _build_messages(
        self,
        sys_prompt: str,
        chat_history: list,
        recommend_item: str,
        generation_mode: str = "gpt_curator",
    ) -> list[dict]:
        fewshot_block = self.fewshot_pool.format_block(self.num_fewshot)
        system_parts = [sys_prompt]
        if fewshot_block:
            system_parts.append(fewshot_block)

        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        for turn in chat_history:
            role = turn.get("role", "user")
            if role not in {"user", "assistant", "system"}:
                role = "user"
            messages.append({"role": role, "content": turn["content"]})

        if generation_mode in {"grounded", "gpt_curator"}:
            user_content = (
                f"{recommend_item}\n\n"
                "Write your assistant reply now. Recommend track #1 from the retrieval block only. "
                "Open by addressing the user's latest request. Name #1's artist and title exactly as shown. "
                "Mention ONLY facts present in the metadata — never invent instrumentation, tempo, or accolades. "
                "Use fresh wording, vary sentence openings, and stay under the requested length."
            )
        else:
            user_content = recommend_item
        messages.append({"role": "user", "content": user_content})
        return messages

    def _complete(self, messages: list[dict], max_new_tokens: int) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    max_tokens=max_new_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as error:
                last_error = error
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"OpenAI generation failed after retries: {last_error}")

    def response_generation(
        self,
        sys_prompt: str,
        chat_history: list,
        recommend_item: str,
        max_new_tokens: int = 160,
        response_format=None,
        generation_mode: str = "gpt_curator",
    ) -> str:
        messages = self._build_messages(
            sys_prompt,
            chat_history,
            recommend_item,
            generation_mode=generation_mode,
        )
        return self._complete(messages, max_new_tokens)

    def batch_response_generation(
        self,
        sys_prompts: list[str],
        chat_histories: list[list],
        recommend_items: list[str],
        max_new_tokens: int = 160,
        generation_mode: str = "gpt_curator",
    ) -> list[str]:
        return [
            self.response_generation(
                sys_prompt,
                chat_history,
                recommend_item,
                max_new_tokens=max_new_tokens,
                generation_mode=generation_mode,
            )
            for sys_prompt, chat_history, recommend_item in zip(
                sys_prompts, chat_histories, recommend_items
            )
        ]
