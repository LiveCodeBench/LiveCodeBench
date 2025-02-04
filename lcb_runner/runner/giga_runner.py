from time import sleep

try:
    from giga import GigaChat
except ImportError as e:
    pass

from requests.exceptions import RequestException

from lcb_runner.runner.base_runner import BaseRunner


class GigaRunner(BaseRunner):
    client = GigaChat()

    def __init__(self, args, model):
        super().__init__(args, model)
        self.client_kwargs: dict[str | str] = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
        }

    def _run_single(self, prompt: list[dict[str, str]]) -> list[str]:
        assert isinstance(prompt, list)

        try:
            response = GigaRunner.client.chat(
                messages=prompt,
                **self.client_kwargs,
            )
        except RequestException as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return [_["message"]["content"] for _ in response["choices"]]
