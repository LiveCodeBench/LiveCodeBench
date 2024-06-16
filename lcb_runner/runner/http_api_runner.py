import openai
from multiprocessing import Pool
from lcb_runner.runner.base_runner import BaseRunner

class HttpAPIRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.model_name = model.model_name

        self.client_kwargs: dict[str | str] = {
            "model": self.model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": args.n,
            "timeout": args.openai_timeout,
        }

        openai.api_base = args.api_url
        openai.api_key = args.api_key


    def get_response(self, prompt):
        try:
            completion = openai.ChatCompletion.create(
                messages=[{"role": "user", "content": prompt}],
                **self.client_kwargs
            )
            results = [choice['message']['content'] for choice in completion.choices]
            return results
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
            time.sleep(30)
            return self.get_response(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e

    def run_single_prompt(self, prompt):
        return self.get_response(prompt)

    def run(self, prompts):
        all_responses = []
        with Pool(processes=len(prompts)) as pool:
            all_responses = pool.map(self.run_single_prompt, prompts)
#        for prompt in prompts:
#            responses = self.get_response(prompt)
#            all_responses.append(responses)
        return all_responses

