from abc import ABC, abstractmethod

from tqdm import tqdm

from lcb_runner.lm_styles import LanguageModel
from lcb_runner.utils.multiprocess import run_tasks_in_parallel

class BaseRunner(ABC):
    def __init__(self, args, model: LanguageModel):
        self.args = args
        self.model = model

    @abstractmethod
    def run_single(self, prompt: str) -> str:
        pass

    def run_batch(self, prompts: list[str]) -> list[str]:
        if self.args.multiprocess > 1:
            return run_tasks_in_parallel(self.run_single, prompts, self.args.multiprocess, self.args.timeout, True)
        return [self.run_single(prompt) for prompt in tqdm(prompts)]

    def run_main(self, benchmark: list, format_prompt: callable) -> list:
        prompts = [format_prompt(problem, self.model.model_style) for problem in benchmark]
        return self.run_batch(prompts)
