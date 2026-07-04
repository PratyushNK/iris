import json
import re
from typing import Literal, Type, TypeVar

from llms.llm import LLM, LLMConfig
from pydantic import BaseModel
from groq import AsyncGroq


T = TypeVar("T", bound=BaseModel)

groq_models = Literal[
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]

class GroqLLM(LLM):

    def __init__(self, client: AsyncGroq):
        self._client = client


    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        llm_config: LLMConfig = LLMConfig(model="llama-3.1-8b-instant"),
    ) -> str:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self._invoke_raw(messages, llm_config)


    @staticmethod
    def _compact_schema(schema: Type[BaseModel]) -> str:
        """Build a compact human-readable field listing (no JSON Schema keywords)."""
        lines = ["{"]
        fields = schema.model_fields
        for i, (name, field) in enumerate(fields.items()):
            anno = field.annotation
            if anno is str:
                type_str = "string"
            elif getattr(anno, "__origin__", None) is list:
                inner = getattr(anno, "__args__", (str,))[0]
                inner_str = getattr(inner, "__name__", str(inner))
                type_str = f"list of {inner_str}s"
            else:
                type_str = getattr(anno, "__name__", str(anno))
            required = field.is_required()
            comma = "," if i < len(fields) - 1 else ""
            lines.append(f'  "{name}": {type_str}{" (required)" if required else ""}{comma}')
        lines.append("}")
        return "\n".join(lines)

    async def generate_structured(
        self,
        schema: Type[T],
        prompt: str,
        system_prompt: str | None = None,
        llm_config: LLMConfig = LLMConfig(model="llama-3.1-8b-instant"),
    ) -> T:

        compact = self._compact_schema(schema)
        structured_system = (
            f"{system_prompt}\n\n" if system_prompt else ""
        ) + (
            f"You MUST respond with ONLY a valid JSON object with these fields:\n"
            f"{compact}\n"
            f"No explanation. No markdown. No code blocks. Just the raw JSON object."
        )

        messages = [
            {"role": "system", "content": structured_system},
            {"role": "user", "content": prompt},
        ]

        last_error: str | None = None
        last_raw: str | None = None

        for attempt in range(llm_config.max_retries + 1):
            if attempt > 0 and last_raw is not None:
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid.\n"
                        f"Response: {last_raw}\n"
                        f"Error: {last_error}\n"
                        f"Return ONLY the corrected JSON object."
                    )
                })

            response: str = await self._invoke_raw(messages, llm_config)
            raw_text: str = response
            last_raw = raw_text

            try:
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
                parsed = json.loads(cleaned)
                return schema.model_validate(parsed)

            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                continue

        raise ValueError(
            f"Failed to get valid structured output after {llm_config.max_retries} attempts.\n"
            f"Last raw response: {last_raw}\n"
            f"Last error: {last_error}"
        )


    async def _invoke_raw(
        self,
        messages: list,
        llm_config: LLMConfig,
    ) -> str:
        request_kwargs = {
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "messages": messages,
        }

        if any(
            msg.get("role") == "system" and "valid JSON object" in str(msg.get("content", ""))
            for msg in messages
        ):
            request_kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**request_kwargs)

        return self._build_llm_response(
            response=response,
        )
    
    def _build_llm_response(
        self,
        response,
    ) -> str:

        if not response.choices:
            raise ValueError("Groq returned no choices")

        choice = response.choices[0]

        if choice is None:
            raise ValueError("Groq returned no choice")

        message = choice.message

        if message is None:
            raise ValueError("Groq returned no message")

        content = message.content

        if not isinstance(content, str):
            raise ValueError(f"Unexpected content type from Groq: {type(content)}")

        return content
    