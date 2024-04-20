import json

from lcb_runner.runner.parser import get_args
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    sort_and_extract_save_results,
    get_metrics,
)


def main():
    args = get_args()

    benchmark, _ = build_prompt_benchmark(args)

    with open(args.custom_output_file, "r") as f:
        custom_outputs = json.load(f)
        assert isinstance(custom_outputs, list)
        assert len(custom_outputs) == len(benchmark)
        if isinstance(custom_outputs[0], list):
            ## custom outputs must list[list[str]]
            ## list of extracted outputs per question
            ## sorted by the benchmark question_id

            assert all(
                isinstance(custom_output, list) for custom_output in custom_outputs
            )
        elif isinstance(custom_outputs[0], dict):
            ## custom outputs must list[dict[str, Any]]
            ## list of extracted outputs per question
            ## sorted by the benchmark question_id

            assert all(
                isinstance(custom_output, dict) for custom_output in custom_outputs
            )
            custom_outputs = [
                custom_output["code_list"]
                for custom_output in sorted(
                    custom_outputs, key=lambda x: str(x["question_id"])
                )
            ]

    save_results = [
        instance.insert_output(custom_output, custom_output)
        for instance, custom_output in zip(benchmark, custom_outputs)
    ]

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    output_path = args.custom_output_file[:-5] + f"_{args.scenario.value}_output.json"

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    metrics = get_metrics(args.scenario, args, benchmark, combined_results)
    graded = extract_instance_results(metrics[1])

    save_eval_results = [
        instance.insert_output_evaluation(outputs_list, extracted_list, graded_list)
        for instance, (outputs_list, extracted_list), graded_list in zip(
            benchmark, combined_results, graded
        )
    ]

    with open(output_path.replace(".json", "_eval.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(output_path.replace(".json", "_eval_all.json"), "w") as f:
        json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()
