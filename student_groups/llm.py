from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from urllib import error, request


@dataclass
class LLMResponse:
    text: str
    used_model: bool


class OllamaClient:
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        if system_prompt:
            payload["system"] = system_prompt

        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=12) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError):
            return LLMResponse(text="", used_model=False)

        text = (data.get("response") or "").strip()
        return LLMResponse(text=text, used_model=bool(text))
