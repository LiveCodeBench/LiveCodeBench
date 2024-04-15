import os
import json

from lcb_runner.runner.parser import get_args
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)


def main():
    args = get_args()

    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args.scenario)
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    output_path = get_output_path(model, args)

    if args.continue_existing:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = build_runner(args, model)
        results: list[str] = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []

    combined_results = combine_results(args.scenario, results, model)

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(benchmark, combined_results)
    ]

    if args.continue_existing:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        # TODO: we can add --continue_existing_evaluation flag to continue from existing evaluation
        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])

        save_eval_results = [
            instance.insert_output_evaluation(outputs_list, extracted_list, graded_list)
            for instance, (outputs_list, extracted_list), graded_list in zip(
                benchmark, combined_results, graded
            )
        ]

        metadata = [] 
        for key in sorted(list(metrics[2].keys())):
            metadata.append(metrics[2][key])
        metrics_list = list(metrics)
        del metrics_list[2]
        metrics = tuple(metrics_list)
        for i in range(len(metadata)):
            if type(metadata[i]) is not list:
                metadata[i] = [str(metadata[i])]
            else:
                metadata[i] = [str(x) for x in metadata[i]]

        save_eval_results = [
            instance.insert_output_evaluation(outputs_list, extracted_list, graded_list,meta)
            for instance, (outputs_list, extracted_list), graded_list,meta in zip(
                benchmark, combined_results, graded,metadata
            )
        ]

        with open(output_path.replace(".json", "_eval.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(output_path.replace(".json", "_eval_all.json"), "w") as f:
            json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()
