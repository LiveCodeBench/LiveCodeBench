import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.runner.base_runner import BaseRunner


class GrokRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("GROK_API_KEY"),
        base_url="https://api.x.ai/v1",
    )

    def __init__(self, args, model):
        super().__init__(args, model)
        model_name = args.model.split("_")[0]
        self.client_kwargs: dict[str | str] = {
            "model": model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "n": 1,
            # "timeout": args.openai_timeout,
            # "stop": args.stop, --> stop is only used for base models currently
        }
        if "_" in args.model:
            self.client_kwargs["reasoning_effort"] = args.model.split("_")[1]

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        def __run_single(counter):
            try:
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                content = response.choices[0].message.content
                return content
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
                print(prompt[0]["content"])
                print("Sleeping for 30 seconds...")
                print("Consider reducing the number of parallel processes.")
                sleep(30)
                return GrokRunner._run_single(prompt)
            except Exception as e:
                print(f"Failed to run the model for {prompt}!")
                print("Exception: ", repr(e))
                raise e

        outputs = []
        try:
            for _ in range(self.args.n):
                outputs.append(__run_single(10))
        except Exception as e:
            raise e
        return outputs
