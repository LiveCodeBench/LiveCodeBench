import json

from lcb_runner.evaluation import codegen_metrics
from lcb_runner.runner.vllm_runner import VLLMRunner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.runner.parser import Scenario, get_args
from lcb_runner.prompts import format_prompt_generation
from lcb_runner.lm_styles import LanguageModel, LanguageModelStore, LMStyle
from lcb_runner.benchmarks import CodeGenerationProblem, load_generation_dataset

from lcb_runner.utils.extraction_utils import extract_code


def main():
    args = get_args()

    if args.scenario == Scenario.generation:
        benchmark: list[CodeGenerationProblem] = load_generation_dataset()
        format_prompt = format_prompt_generation
    else:
        raise ValueError(f"Scenario {args.scenario} not implemented")

    if args.continue_existing:
        raise ValueError("Continue existing not implemented")

    model = LanguageModelStore[args.model]
    runner = VLLMRunner(args, model)
    results: list[str] = runner.run_main(benchmark, format_prompt)

    if args.scenario == Scenario.generation:
        combined_results = [
            (
                outputs_list,
                [extract_code(output, model.model_style) for output in outputs_list],
            )
            for outputs_list in results
        ]
    else:
        raise ValueError(f"Scenario {args.scenario} not implemented")

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(benchmark, combined_results)
    ]

    output_path = get_output_path(model, args)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
        generations = [extracted for _, extracted in combined_results]
        metrics = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
        )

        print(metrics[0]["pass@1"])
        with open(output_path.replace(".json", "_eval.json"), "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
