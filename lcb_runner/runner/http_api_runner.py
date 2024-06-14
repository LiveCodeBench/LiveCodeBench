import openai
import time

class HttpAPIRunner:
    def __init__(self, api_url, api_key, model_name, num_samples):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.num_samples = num_samples

    def get_response(self, prompt):
        openai.api_base = self.api_url
        openai.api_key = self.api_key
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            stream=False,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10000,
            n=self.num_samples
        )
        results = [choice['message']['content'] for choice in completion.choices]
        return results

    def run(self, prompts):
        all_responses = []
        for prompt in prompts:
            responses = self.get_response(prompt)
            all_responses.append(responses)
        return all_responses

