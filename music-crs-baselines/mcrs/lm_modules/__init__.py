from .llama import LLAMA_MODEL


def load_lm_module(lm_type, device, attn_implementation, dtype, lm_kwargs=None):
    lm_kwargs = lm_kwargs or {}
    if lm_type == "meta-llama/Llama-3.2-1B-Instruct":
        return LLAMA_MODEL(
            model_name=lm_type,
            device=device,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
    if lm_type in {"gpt-4o-mini", "gpt-4o", "openai/gpt-4o-mini"}:
        from .openai_lm import OpenAILM

        model_name = lm_kwargs.pop("model_name", "gpt-4o-mini")
        return OpenAILM(model_name=model_name, **lm_kwargs)
    raise ValueError(f"Unsupported LM type: {lm_type}")
