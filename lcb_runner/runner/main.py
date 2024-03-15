import json

from lcb_runner.runner.vllm_runner import VLLMRunner
from lcb_runner.runner.parser import Scenario, get_args
from lcb_runner.prompts.generation import format_prompt_generation
from lcb_runner.lm_styles import LanguageModel, LanguageModelStore, LMStyle
from lcb_runner.benchmarks.generation import GenerationProblem, load_generation_dataset

def main():
    args = get_args()


    if args.scenario == Scenario.generation:
        benchmark = load_generation_dataset()
        format_prompt = format_prompt_generation
    else:
        raise ValueError(f"Scenario {args.scenario} not implemented")

    if args.continue_existing:
        raise ValueError("Continue existing not implemented")
    
    model = LanguageModelStore[args.model]
    runner = VLLMRunner(args, model)
    results = runner.run_main(benchmark, format_prompt)

    with open(args.output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()