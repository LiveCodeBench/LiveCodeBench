import json

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.evaluation import codegen_metrics, test_output_metrics
from lcb_runner.runner.vllm_runner import VLLMRunner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.prompts import format_prompt_generation, format_prompt_test_output
from lcb_runner.lm_styles import LanguageModel, LanguageModelStore, LMStyle
from lcb_runner.benchmarks import (
    load_code_generation_dataset,
    load_test_prediction_dataset,
)
from lcb_runner.utils.extraction_utils import extract_code, extract_test_output_code


def main():
    args = get_args()

    if args.scenario == Scenario.codegeneration:
        benchmark = load_code_generation_dataset()
        format_prompt = format_prompt_generation
    elif args.scenario == Scenario.testoutputprediction:
        benchmark = load_test_prediction_dataset()
        format_prompt = format_prompt_test_output
    else:
        raise ValueError(f"Scenario {args.scenario} not implemented")

    if args.continue_existing:
        raise ValueError("Continue existing not implemented")

    model = LanguageModelStore[args.model]
    runner = VLLMRunner(args, model)
    results: list[str] = runner.run_main(benchmark, format_prompt)

    if args.scenario == Scenario.codegeneration:
        combined_results = [
            (
                outputs_list,
                [extract_code(output, model.model_style) for output in outputs_list],
            )
            for outputs_list in results
        ]
    elif args.scenario == Scenario.testoutputprediction:
        combined_results = [
            (
                outputs_list,
                [
                    extract_test_output_code(output, model.model_style)
                    for output in outputs_list
                ],
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
        if args.scenario == Scenario.codegeneration:
            eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
            generations = [extracted for _, extracted in combined_results]
            metrics = codegen_metrics(
                eval_samples,
                generations,
                num_process_evaluate=args.num_process_evaluate,
                timeout=args.timeout,
            )

            print(metrics[0]["pass@1"])

        elif args.scenario == Scenario.testoutputprediction:
            eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
            generations = [extracted for _, extracted in combined_results]
            metrics = test_output_metrics(
                eval_samples,
                generations,
                k_list=[1, 5],
            )

            print(metrics[0]["pass@1"])

        with open(output_path.replace(".json", "_eval.json"), "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
