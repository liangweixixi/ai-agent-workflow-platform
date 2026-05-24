"""多模型适配层 - 统一接口调用 Claude / MiMo / DeepSeek / GPT"""

import os
from dataclasses import dataclass
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    name: str
    provider: str
    base_url: str
    api_key: str
    model_id: str
    max_tokens: int = 4096


def _get_model_configs() -> dict[str, ModelConfig]:
    return {
        # Xiaomi MiMo
        "mimo-v2.5-pro": ModelConfig(
            name="mimo-v2.5-pro",
            provider="mimo",
            base_url=os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
            api_key=os.getenv("MIMO_API_KEY", ""),
            model_id="MiMo-V2.5-Pro",
            max_tokens=8192,
        ),
        "mimo-v2.5": ModelConfig(
            name="mimo-v2.5",
            provider="mimo",
            base_url=os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
            api_key=os.getenv("MIMO_API_KEY", ""),
            model_id="MiMo-V2.5",
            max_tokens=8192,
        ),
        # DeepSeek
        "deepseek-chat": ModelConfig(
            name="deepseek-chat",
            provider="deepseek",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model_id="deepseek-chat",
            max_tokens=4096,
        ),
        "deepseek-reasoner": ModelConfig(
            name="deepseek-reasoner",
            provider="deepseek",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model_id="deepseek-reasoner",
            max_tokens=8192,
        ),
        # OpenAI GPT
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            provider="openai",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_id="gpt-4o",
            max_tokens=4096,
        ),
    }


class ModelManager:
    def __init__(self):
        self._configs = _get_model_configs()
        self._client = httpx.AsyncClient(timeout=120.0)

    def list_models(self) -> list[dict]:
        return [
            {"name": cfg.name, "provider": cfg.provider, "max_tokens": cfg.max_tokens}
            for cfg in self._configs.values()
        ]

    def get_config(self, model_name: str) -> ModelConfig:
        if model_name not in self._configs:
            raise ValueError(f"未知模型: {model_name}，可用模型: {list(self._configs.keys())}")
        return self._configs[model_name]

    async def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        config = self.get_config(model)

        if config.provider == "anthropic":
            return await self._call_anthropic(config, messages, temperature, max_tokens)
        else:
            return await self._call_openai_compatible(config, messages, temperature, max_tokens)

    async def _call_openai_compatible(
        self,
        config: ModelConfig,
        messages: list[dict],
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or config.max_tokens,
        }

        response = await self._client.post(
            f"{config.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _call_anthropic(
        self,
        config: ModelConfig,
        messages: list[dict],
        temperature: float,
        max_tokens: Optional[int],
    ) -> str:
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n"
            else:
                chat_messages.append(msg)

        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.model_id,
            "max_tokens": max_tokens or config.max_tokens,
            "temperature": temperature,
            "system": system_msg.strip(),
            "messages": chat_messages,
        }

        response = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

    async def close(self):
        await self._client.aclose()
