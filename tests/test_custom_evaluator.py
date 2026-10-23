"""Offline regression tests exercising the real custom_evaluator.main.

Only dataset loading, CLI parsing, and metric execution are mocked. No model,
dataset download, or generated program is executed. Run with LCB_REPO set.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO = Path(os.environ.get("LCB_REPO", Path(__file__).resolve().parents[1])).resolve()


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CustomEvaluatorRegression(unittest.TestCase):
    def run_case(self, custom_outputs, scenario_name, expected):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "answers.json"
            source.write_text(json.dumps(custom_outputs))
            scenario = load_source(
                "audit_scenarios", REPO / "lcb_runner/utils/scenarios.py"
            ).Scenario
            records = []
            for idx in range(len(expected)):

                def insert_output(raw, predictions, idx=idx):
                    return {
                        "id": f"row_{idx}",
                        "output_list": raw,
                        "pred_list": predictions,
                    }

                def insert_evaluation(raw, predictions, grades):
                    return {"pred_list": predictions, "graded_list": grades}

                records.append(
                    SimpleNamespace(
                        insert_output=insert_output,
                        insert_output_evaluation=insert_evaluation,
                    )
                )
            args = SimpleNamespace(
                scenario=getattr(scenario, scenario_name),
                custom_output_file=str(source),
                custom_output_save_name=None,
            )
            received = []

            def sort_extract(current_scenario, outputs):
                received.extend(output["pred_list"] for output in outputs)
                return outputs, [(x["output_list"], x["pred_list"]) for x in outputs]

            modules = {}
            for name in [
                "lcb_runner.runner.parser",
                "lcb_runner.utils.scenarios",
                "lcb_runner.utils.path_utils",
                "lcb_runner.evaluation",
                "lcb_runner.runner.scenario_router",
            ]:
                modules[name] = ModuleType(name)
            modules["lcb_runner.runner.parser"].get_args = lambda: args
            modules["lcb_runner.utils.scenarios"].Scenario = scenario
            modules["lcb_runner.utils.path_utils"].get_output_path = lambda _: str(
                source
            )
            modules["lcb_runner.evaluation"].extract_instance_results = lambda _: [
                [True] for _ in records
            ]
            router = modules["lcb_runner.runner.scenario_router"]
            router.build_prompt_benchmark = lambda _: (records, None)
            router.sort_and_extract_save_results = sort_extract
            router.get_metrics = lambda *args: [{"pass@1": 1.0}, {}]
            with patch.dict(sys.modules, modules):
                evaluator = load_source(
                    "audit_custom_evaluator",
                    REPO / "lcb_runner/runner/custom_evaluator.py",
                )
                evaluator.main()
            self.assertEqual(received, expected)
            saved = json.loads(
                source.with_name(
                    f"answers_{scenario_name}_output_eval_all.json"
                ).read_text()
            )
            self.assertEqual([row["pred_list"] for row in saved], expected)

    def test_codeexecution_dictionary_input_numeric_order(self):
        self.run_case(
            [
                {"id": "row_10", "pred_list": ["10"]},
                {"id": "row_2", "pred_list": ["2"]},
            ],
            "codeexecution",
            [["2"], ["10"]],
        )

    def test_codeexecution_list_input_unchanged(self):
        self.run_case([["2"], ["10"]], "codeexecution", [["2"], ["10"]])

    def test_testoutputprediction_dictionary_order_unchanged(self):
        self.run_case(
            [
                {"question_id": "b", "test_id": "1", "pred_list": ["b"]},
                {"question_id": "a", "test_id": "1", "pred_list": ["a"]},
            ],
            "testoutputprediction",
            [["a"], ["b"]],
        )


if __name__ == "__main__":
    unittest.main()
