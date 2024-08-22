from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LMStyle(Enum):
    OpenAIChat = "OpenAIChat"
    Claude = "Claude"  # Claude 1 and Claude 2
    Claude3 = "Claude3"
    Gemini = "Gemini"
    MistralWeb = "MistralWeb"
    CohereCommand = "CohereCommand"
    DataBricks = "DataBricks"
    DeepSeekAPI = "DeepSeekAPI"

    GenericBase = "GenericBase"

    DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
    CodeLLaMaInstruct = "CodeLLaMaInstruct"
    StarCoderInstruct = "StarCoderInstruct"
    CodeQwenInstruct = "CodeQwenInstruct"

    Phind = "Phind"
    WizardCoder = "WizardCoder"
    MagiCoder = "MagiCoder"
    OC = "OC"
    Eurusx = "Eurusx"

    Qwen1point5 = "Qwen1point5"
    Smaug2 = "Smaug2"

    LLaMa2 = "LLaMa2"
    LLaMa3 = "LLaMa3"
    Mistral = "Mistral"
    Yi = "Yi"


@dataclass
class LanguageModel:
    model_name: str
    model_repr: str
    model_style: LMStyle
    release_date: datetime | None  # XXX Should we use timezone.utc?
    link: str | None = None

    def __hash__(self) -> int:
        return hash(self.model_name)


LanguageModelList: list[LanguageModel] = [
    LanguageModel(
        "meta-llama/Llama-2-7b-chat-hf",
        "Llama-2-7b-chat",
        LMStyle.LLaMa2,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
    ),
    LanguageModel(
        "meta-llama/Llama-2-13b-chat-hf",
        "Llama-2-13b-chat",
        LMStyle.LLaMa2,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Llama-2-13b-chat-hf",
    ),
    LanguageModel(
        "meta-llama/Llama-2-70b-chat-hf",
        "Llama-2-70b-chat",
        LMStyle.LLaMa2,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-70B",
        "LLama3-70b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-70B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-8B",
        "LLama3-8b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-8B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "LLama3-8b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "LLama3-70b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-8B",
        "LLama3.1-8b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-70B",
        "LLama3.1-70b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-405B-FP8",
        "LLama3.1-405b-Base-FP8",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "LLama3.1-8b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "LLama3.1-70b-Ins",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    LanguageModel(
        "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        "LLama3.1-405b-Ins-FP8",
        LMStyle.LLaMa3,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-33b-base",
        "DSCoder-33b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-33b-base",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-6.7b-base",
        "DSCoder-6.7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-1.3b-base",
        "DSCoder-1.3b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-base",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-33b-instruct",
        "DSCoder-33b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 9, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "DSCoder-6.7b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 9, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct",
    ),
    LanguageModel(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "DSCoder-1.3b-Ins",
        LMStyle.DeepSeekCodeInstruct,
        datetime(2023, 8, 1),
        link="https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct",
    ),
    LanguageModel(
        "deepseek-chat",
        "DeepSeek-V2",
        LMStyle.DeepSeekAPI,
        datetime(2023, 8, 1),
        link="https://huggingface.co/deepseek-ai/DeepSeek-V2",
    ),
    LanguageModel(
        "deepseek-coder",
        "DeepSeekCoder-V2",
        LMStyle.DeepSeekAPI,
        datetime(2023, 8, 1),
        link="https://huggingface.co/deepseek-ai/DeepSeek-V2",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-70b-hf",
        "CodeLlama-70b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-70b-hf",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-34b-hf",
        "CodeLlama-34b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-34b-hf",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-13b-hf",
        "CodeLlama-13b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-13b-hf",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-7b-hf",
        "CodeLlama-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-7b-hf",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-70b-Instruct-hf",
        "CodeLlama-70b-Ins",
        LMStyle.CodeLLaMaInstruct,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-70b-hf",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-34b-Instruct-hf",
        "CodeLlama-34b-Ins",
        LMStyle.CodeLLaMaInstruct,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-34b-Instruct-hf",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-13b-Instruct-hf",
        "CodeLlama-13b-Ins",
        LMStyle.CodeLLaMaInstruct,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf",
    ),
    LanguageModel(
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "CodeLlama-7b-Ins",
        LMStyle.CodeLLaMaInstruct,
        datetime(2023, 1, 1),
        link="https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf",
    ),
    LanguageModel(
        "gpt-3.5-turbo-0301",
        "GPT-3.5-Turbo-0301",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
    ),
    LanguageModel(
        "gpt-3.5-turbo-0125",
        "GPT-3.5-Turbo-0125",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://openai.com/blog/new-embedding-models-and-api-updates#:~:text=Other%20new%20models%20and%20lower%20pricing",
    ),
    LanguageModel(
        "gpt-4-0613",
        "GPT-4-0613",
        LMStyle.OpenAIChat,
        datetime(2021, 10, 1),
        link="https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
    ),
    LanguageModel(
        "gpt-4-1106-preview",
        "GPT-4-Turbo-1106",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
    ),
    LanguageModel(
        "gpt-4-turbo-2024-04-09",
        "GPT-4-Turbo-2024-04-09",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
    ),
    LanguageModel(
        "gpt-4o-2024-05-13",
        "GPT-4O-2024-05-13",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "gpt-4o-mini-2024-07-18",
        "GPT-4O-mini-2024-07-18",
        LMStyle.OpenAIChat,
        datetime(2023, 4, 30),
        link="https://openai.com/index/spring-update",
    ),
    LanguageModel(
        "claude-instant-1",
        "Claude-Instant-1",
        LMStyle.Claude,
        datetime(2022, 12, 31),
        link="https://www.anthropic.com/index/introducing-claude",
    ),
    LanguageModel(
        "claude-2",
        "Claude-2",
        LMStyle.Claude,
        datetime(2022, 12, 31),
        link="https://www.anthropic.com/index/claude-2",
    ),
    LanguageModel(
        "claude-3-opus-20240229",
        "Claude-3-Opus",
        LMStyle.Claude3,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-sonnet-20240229",
        "Claude-3-Sonnet",
        LMStyle.Claude3,
        datetime(2023, 9, 1),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "claude-3-5-sonnet-20240620",
        "Claude-3.5-Sonnet",
        LMStyle.Claude3,
        datetime(2024, 3, 31),
        link="https://www.anthropic.com/news/claude-3-5-sonnet",
    ),
    LanguageModel(
        "claude-3-haiku-20240307",
        "Claude-3-Haiku",
        LMStyle.Claude3,
        datetime(2023, 4, 30),
        link="https://www.anthropic.com/index/claude-3",
    ),
    LanguageModel(
        "gemini-pro",
        "Gemini-Pro",
        LMStyle.Gemini,
        datetime(2023, 5, 1),
        link="https://blog.Gemini/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "gemini-1.5-pro-undefined",
        "Gemini-Pro-1.5-April (n=1)",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "gemini-1.5-pro-latest",
        "Gemini-Pro-1.5-May",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "gemini-1.5-pro-exp-0801",
        "Gemini-Pro-1.5-August-True",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "gemini-1.5-flash-latest",
        "Gemini-Flash-1.5-May",
        LMStyle.Gemini,
        datetime(2023, 4, 30),
        link="https://blog.google/technology/ai/gemini-api-developers-cloud",
    ),
    LanguageModel(
        "databricks-dbrx-instruct",
        "DBRX-Ins",
        LMStyle.DataBricks,
        datetime(2023, 1, 1),
        link="https://huggingface.co/databricks/dbrx-instruct",
    ),
    LanguageModel(
        "bigcode/starcoder2-3b",
        "StarCoder2-3b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "bigcode/starcoder2-7b",
        "StarCoder2-7b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "bigcode/starcoder2-15b",
        "StarCoder2-15b",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/bigcode/starcoder2-7b-magicoder-instruct/tree/main",
    ),
    LanguageModel(
        "google/codegemma-7b",
        "CodeGemma-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/codegemma-7b",
    ),
    LanguageModel(
        "google/codegemma-2b",
        "CodeGemma-2b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/codegemma-2b",
    ),
    LanguageModel(
        "google/gemma-7b",
        "Gemma-7b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/gemma-7b",
    ),
    LanguageModel(
        "google/gemma-2b",
        "Gemma-2b-Base",
        LMStyle.GenericBase,
        datetime(2023, 1, 1),
        link="https://huggingface.co/google/gemma-2b",
    ),
    LanguageModel(
        "mistralai/Mistral-7B-Instruct-v0.1",
        "Mistral-7b-Ins-v0.1",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1",
    ),
    LanguageModel(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Mistral-7b-Ins-v0.2",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
    ),
    LanguageModel(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral-7b-Ins-v0.3",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
    ),
    LanguageModel(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Mistral-8x7B-Ins-v0.1",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
    LanguageModel(
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Mistral-8x22B-Ins-v0.1",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1",
    ),
    LanguageModel(
        "mistralai/Mistral-Nemo-Instruct-2407",
        "Mistral-Nemo-Ins-2407",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407",
    ),
    LanguageModel(
        "mistralai/Mistral-Large-Instruct-2407",
        "Mistral-Large-Ins-2407",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407",
    ),
    LanguageModel(
        "mistralai/Codestral-22B-v0.1",
        "Codestral-22B-v0.1",
        LMStyle.Mistral,
        datetime(2023, 1, 1),
        link="https://huggingface.co/mistralai/Codestral-22B-v0.1",
    ),
    LanguageModel(
        "mistral-large-latest",
        "Mistral-Large",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mistral-large/",
    ),
    LanguageModel(
        "open-mixtral-8x22b",
        "Mixtral-8x22B-Ins",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mixtral-8x22b/",
    ),
    LanguageModel(
        "open-mixtral-8x7b",
        "Mixtral-8x7B-Ins",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mixtral-8x7b/",
    ),
    LanguageModel(
        "open-mixtral-8x7b",
        "Mixtral-8x7B-Ins",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/mixtral-8x7b/",
    ),
    LanguageModel(
        "codestral-latest",
        "Codestral-Latest",
        LMStyle.MistralWeb,
        datetime(2023, 1, 1),
        link="https://mistral.ai/news/codestral/",
    ),
    LanguageModel(
        "command-r",
        "Command-R",
        LMStyle.CohereCommand,
        datetime(2023, 1, 1),
        link="https://docs.cohere.com/docs/models",
    ),
    LanguageModel(
        "command-r-plus",
        "Command-R+",
        LMStyle.CohereCommand,
        datetime(2023, 1, 1),
        link="https://docs.cohere.com/docs/models",
    ),
    LanguageModel(
        "Qwen/CodeQwen1.5-7B",
        "CodeQwen15-7B",
        LMStyle.GenericBase,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/CodeQwen1.5-7B",
    ),
    LanguageModel(
        "Qwen/CodeQwen1.5-7B-Chat",
        "CodeQwen15-7B-Chat",
        LMStyle.CodeQwenInstruct,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/CodeQwen1.5-7B-Chat",
    ),
    LanguageModel(
        "Qwen/Qwen2-72B",
        "Qwen2-Base-72B",
        LMStyle.GenericBase,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2-72B",
    ),
    LanguageModel(
        "Qwen/Qwen2-72B-Instruct",
        "Qwen2-Ins-72B",
        LMStyle.CodeQwenInstruct,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2-72B-Instruct",
    ),
    LanguageModel(
        "Qwen/Qwen2-7B",
        "Qwen2-Base-7B",
        LMStyle.GenericBase,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2-7B",
    ),
    LanguageModel(
        "Qwen/Qwen2-7B-Instruct",
        "Qwen2-Ins-7B",
        LMStyle.CodeQwenInstruct,
        datetime(2023, 8, 30),
        link="https://huggingface.co/Qwen/Qwen2-7B-Instruct",
    ),
    LanguageModel(
        "m-a-p/OpenCodeInterpreter-DS-33B",
        "OC-DS-33B",
        LMStyle.OC,
        datetime(2023, 1, 1),
        link="https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-33B/",
    ),
    LanguageModel(
        "m-a-p/OpenCodeInterpreter-DS-6.7B",
        "OC-DS-6.7B",
        LMStyle.OC,
        datetime(2023, 9, 1),
        link="https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-6.7B/",
    ),
    LanguageModel(
        "m-a-p/OpenCodeInterpreter-DS-1.3B",
        "OC-DS-1.3B",
        LMStyle.OC,
        datetime(2023, 9, 1),
        link="https://huggingface.co/m-a-p/OpenCodeInterpreter-DS-1.3B/",
    ),
    LanguageModel(
        "stabilityai/stable-code-3b",
        "StableCode-3B",
        LMStyle.GenericBase,
        datetime(2023, 9, 1),
        link="https://huggingface.co/stabilityai/stable-code-3b/",
    ),
    LanguageModel(
        "bigcode/starcoder2-instruct-15b-v0.1",
        "StarCoder2-Ins-v0.1",
        LMStyle.LLaMa3,
        datetime(2023, 4, 30),
        link="https://huggingface.co/bigcode/starcoder2-instruct-15b-v0.1",
    ),
    LanguageModel(
        "qwen/Qwen1.5-72B-Chat",
        "Qwen-1.5-72B-Chat ",
        LMStyle.Qwen1point5,
        datetime(2024, 3, 31),
        link="https://huggingface.co/qwen/Qwen1.5-72B-Chat/",
    ),
    LanguageModel(
        "abacusai/Smaug-2-72B",
        "Smaug-2-72B ",
        LMStyle.Smaug2,
        datetime(2024, 3, 31),
        link="https://huggingface.co/abacusai/Smaug-2-72B/",
    ),
    LanguageModel(
        "WizardCoderLM/WizardCoderCoder-Python-34B-V1.0",
        "WCoder-34B-V1",
        LMStyle.WizardCoder,
        datetime(2023, 1, 1),
        link="https://huggingface.co/WizardCoderLM/WizardCoderCoder-Python-34B-V1.0",
    ),
    LanguageModel(
        "WizardCoderLM/WizardCoderCoder-33B-V1.1",
        "WCoder-33B-V1.1",
        LMStyle.WizardCoder,
        datetime(2023, 9, 1),
        link="https://huggingface.co/WizardCoderLM/WizardCoderCoder-33B-V1.1",
    ),
    LanguageModel(
        "Phind/Phind-CodeLlama-34B-v2",
        "Phind-34B-V2",
        LMStyle.Phind,
        datetime(2023, 1, 1),
        link="https://huggingface.co/Phind/Phind-CodeLlama-34B-v2",
    ),
    LanguageModel(
        "ise-uiuc/Magicoder-S-DS-6.7B",
        "MagiCoderS-DS-6.7B",
        LMStyle.MagiCoder,
        datetime(2023, 7, 30),
        link="https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B",
    ),
    LanguageModel(
        "ise-uiuc/Magicoder-S-CL-7B",
        "MagiCoderS-CL-7B",
        LMStyle.MagiCoder,
        datetime(2023, 1, 1),
        link="https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B",
    ),
    LanguageModel(
        "openbmb/Eurus-70b-sft",
        "Eurus-70B-SFT (n=1)",
        LMStyle.Eurusx,
        datetime(2023, 1, 1),
        link="https://huggingface.co/openbmb/Eurus-70b-sft",
    ),
    LanguageModel(
        "openbmb/Eurux-8x22b-nca",
        "Eurux-8x22b-NCA (n=1)",
        LMStyle.Eurusx,
        datetime(2023, 4, 30),
        link="https://huggingface.co/openbmb/Eurux-8x22b-nca",
    ),
    LanguageModel(
        "01-ai/Yi-1.5-6B-Chat",
        "Yi-1.5-6B-Chat",
        LMStyle.Yi,
        datetime(2023, 1, 1),
        link="https://huggingface.co/01-ai/Yi-1.5-6B-Chat",
    ),
    LanguageModel(
        "01-ai/Yi-1.5-9B-Chat",
        "Yi-1.5-9B-Chat",
        LMStyle.Yi,
        datetime(2023, 1, 1),
        link="https://huggingface.co/01-ai/Yi-1.5-9B-Chat",        
    ),
    LanguageModel(
        "01-ai/Yi-1.5-34B-Chat",
        "Yi-1.5-34B-Chat",
        LMStyle.Yi,
        datetime(2023, 1, 1),
        link="https://huggingface.co/01-ai/Yi-1.5-34B-Chat",
    ),
]

LanguageModelStore: dict[str, LanguageModel] = {
    lm.model_name: lm for lm in LanguageModelList
}

if __name__ == "__main__":
    print(list(LanguageModelStore.keys()))
