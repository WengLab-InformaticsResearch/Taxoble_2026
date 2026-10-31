import os
from types import SimpleNamespace


def _temperature_for_attempt(
    temperature,
    retry_temperature,
    attempt_number,
):
    return retry_temperature if attempt_number > 1 else temperature


class LangChainChatAdapter:
    """Apply the configured retry temperature to a LangChain chat model."""

    def __init__(self, model, temperature=0.0, retry_temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.retry_temperature = retry_temperature

    def invoke(self, prompt, attempt_number=1):
        effective_temperature = _temperature_for_attempt(
            self.temperature,
            self.retry_temperature,
            attempt_number,
        )
        return self.model.bind(temperature=effective_temperature).invoke(prompt)


class OpenAIChatAdapter:
    """Expose raw OpenAI responses through the `.invoke()` interface used here."""

    def __init__(
        self,
        model,
        api_key=None,
        temperature=0.0,
        retry_temperature=0.1,
        timeout=60,
        max_output_tokens=4096,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Set cfg.llm.api_key or OPENAI_API_KEY before using OpenAI models."
            )

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.retry_temperature = retry_temperature
        self.max_output_tokens = max_output_tokens

    def invoke(self, prompt, attempt_number=1):
        effective_temperature = _temperature_for_attempt(
            self.temperature,
            self.retry_temperature,
            attempt_number,
        )
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=effective_temperature,
            max_output_tokens=self.max_output_tokens,
        )
        usage = response.usage
        return SimpleNamespace(
            content=response.output_text,
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            temperature=effective_temperature,
        )


class VLLMChatAdapter:
    """Expose a vLLM OpenAI-compatible server through `.invoke()`."""

    def __init__(
        self,
        model,
        base_url="http://localhost:8000/v1",
        api_key=None,
        temperature=0.0,
        retry_temperature=0.1,
        top_p=1.0,
        seed=0,
        timeout=120,
        max_completion_tokens=8192,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.retry_temperature = retry_temperature
        self.top_p = top_p
        self.seed = seed
        self.max_completion_tokens = max_completion_tokens

    def seed_for_attempt(self, attempt_number=1):
        return (
            None
            if self.seed is None
            else self.seed + max(attempt_number - 1, 0)
        )

    def invoke(self, prompt, attempt_number=1):
        effective_seed = self.seed_for_attempt(attempt_number)
        effective_temperature = _temperature_for_attempt(
            self.temperature,
            self.retry_temperature,
            attempt_number,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=effective_temperature,
            top_p=self.top_p,
            seed=effective_seed,
            max_completion_tokens=self.max_completion_tokens,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        return SimpleNamespace(
            content=content,
            seed=effective_seed,
            finish_reason=response.choices[0].finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            temperature=effective_temperature,
        )


def invoke_chat_model(model, prompt, attempt_number=1):
    """Invoke a chat model with retry sampling settings when supported."""

    if isinstance(
        model,
        (LangChainChatAdapter, OpenAIChatAdapter, VLLMChatAdapter),
    ):
        return model.invoke(prompt, attempt_number=attempt_number)
    return model.invoke(prompt)


def build_chat_model(cfg):
    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama

        return LangChainChatAdapter(
            ChatOllama(
                base_url=cfg.base_url,
                model=cfg.model,
                temperature=cfg.temperature,
                client_kwargs={"timeout": cfg.timeout},
            ),
            temperature=cfg.temperature,
            retry_temperature=cfg.retry_temperature,
        )

    if cfg.provider == "openai":
        return OpenAIChatAdapter(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            retry_temperature=cfg.retry_temperature,
            timeout=cfg.timeout,
            max_output_tokens=cfg.max_output_tokens,
        )

    if cfg.provider == "vllm":
        return VLLMChatAdapter(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            retry_temperature=cfg.retry_temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
            timeout=cfg.timeout,
            max_completion_tokens=cfg.max_output_tokens,
        )

    raise ValueError(f"Unsupported LLM provider: {cfg.provider}")
