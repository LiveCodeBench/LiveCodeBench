try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Cannot import vllm")

from lcb_runner.runner.base_runner import BaseRunner

class VLLMRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.llm = LLM(model.model_name)
        self.sampling_params = SamplingParams(
                n=self.args.n,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=self.args.stop,
        )

    def run_single(self, prompt: str) -> list[str]:
        return self.llm.generate(prompt, self.sampling_params)
    
    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        return self.llm.generate(prompts, self.sampling_params)