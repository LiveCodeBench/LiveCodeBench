from lcb_runner.lm_styles import LMStyle, LanguageModel
from lcb_runner.runner.vllm_runner import VLLMRunner
from lcb_runner.runner.oai_runner import OpenAIRunner


def build_runner(args, model: LanguageModel) -> OpenAIRunner | VLLMRunner:
    if model.model_style == LMStyle.OpenAIChat:
        return OpenAIRunner(args, model)
    elif model.model_style in [
        LMStyle.Anthropic,
        LMStyle.AnthropicMessage,
        LMStyle.Gemini,
        LMStyle.MistralWeb,
    ]:
        raise NotImplementedError(
            f"Runner for language model style {model.model_style} not implemented yet"
        )
    else:
        return VLLMRunner(args, model)
