import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class OpenAIRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        # Initialize client with custom base URL if provided
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            # Only check for reasoning effort if using OpenAI's official models
            if base_url == "https://api.openai.com/v1":
                assert (
                    "__" in args.model
                ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
                model, reasoning_effort = args.model.split("__")
                self.client_kwargs: dict[str | str] = {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                }
            else:
                self.client_kwargs: dict[str | str] = {
                    "model": args.model,
                }
        elif model.model_style == LMStyle.OpenAIGeneric:
            # For generic OpenAI-compatible endpoints, use minimal configuration
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "n": args.n,
                "timeout": args.openai_timeout,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
            }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        try:
            response = self.client.chat.completions.create(
                messages=prompt,
                **self.client_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [c.message.content for c in response.choices]
