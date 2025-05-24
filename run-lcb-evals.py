#!/usr/bin/env python3

# This script runs the LiveCodeBench custom evaluator for each of the models
# listed in `MODELS`.
#
# It assumes that post-processed data in the format expected by the LiveCodeBench
# custom evaluator: https://github.com/LiveCodeBench/LiveCodeBench?tab=readme-ov-file#custom-evaluation)
# is stored in the following directory structure:
#
# data/
# ├─ gpt-4.1-2025-04-14/
# │  ├─ post-processed/
# │  │  ├─ lcb-formatted.json
# ├─ Llama-3.3-70B-Instruct/
# │  ├─ post-processed/
# │  │  ├─ lcb-formatted.json
#
# Ensure you are running this script in a directory that also contains the `data`
# folder at the same depth.
#
# Example usage:
#   ./run-lcb-evals.py

from dataclasses import dataclass

import glob
import subprocess
import concurrent.futures


@dataclass
class LcbModelEvaluationInfo:
    model: str
    lcb_input_files: list[str]


def main() -> None:
    model_to_lcb_input_files: dict[str, list[str]] = {
        model: glob.glob(f"data/{model}/post-processed/*.json") for model in MODELS
    }
    model_evaluation_infos: list[LcbModelEvaluationInfo] = [
        LcbModelEvaluationInfo(model=model, lcb_input_files=lcb_inputs)
        for model, lcb_inputs in model_to_lcb_input_files.items()
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
        future_to_model = {
            executor.submit(_run_lcb, lcb_evaluation_info): lcb_evaluation_info.model
            for lcb_evaluation_info in model_evaluation_infos
        }

        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                future.result()
            except Exception as e:
                print(
                    f"LiveCodeBench run for model: {model} raised an exception: {str(e)}"
                )


MODELS = [
    "codellama-70b",
    "Codestral-2501",
    "DeepSeek-R1",
    "DeepSeek-V3",
    "gpt-4.1-2025-04-14",
    "Llama-3.3-70B-Instruct",
    "qwen-2.5-coder",
]


def _run_lcb(model_eval_info: LcbModelEvaluationInfo) -> None:
    """Run the LiveCodeBench custom evaluator given LiveCodeBench model evaluation
    information.

    Args:
        model_eval_info (LcbModelEvaluationInfo): The LiveCodeBench model evaluation
        information.
    """
    print(f"Running LiveCodeBench for: {model_eval_info.model}")
    for lcb_input_file in model_eval_info.lcb_input_files:
        run_lcb_for_model = f"python -m lcb_runner.runner.custom_evaluator --custom_output_file {lcb_input_file}"
        subprocess.run(
            run_lcb_for_model,
            shell=True,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
