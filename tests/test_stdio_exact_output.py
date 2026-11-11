import json
import unittest

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness


def problem_sample(expected, metadata, *, test_type="stdin", input_text=""):
    problem = CodeGenerationProblem(
        question_title="Output format fixture",
        question_content="",
        platform="leetcode" if test_type == "functional" else "atcoder",
        question_id="fixture",
        contest_id="fixture",
        contest_date="2026-01-01",
        starter_code="",
        difficulty="easy",
        public_test_cases=json.dumps(
            [{"input": input_text, "output": expected, "testtype": test_type}]
        ),
        private_test_cases="[]",
        metadata=json.dumps(metadata),
    )
    return problem.get_evaluation_sample()


class ExactStdioOutputTests(unittest.TestCase):
    def test_required_format_rejects_equivalent_numeric_spellings(self):
        pairs = [
            ("0011", "11"),
            ("5", "5.0"),
            ("5", "+5"),
            ("10", "1e1"),
            ("10", "1_0"),
            ("0", "-0"),
            ("3.14159265358979323846264338327950", "3.1415926535897932384626433832795"),
        ]
        for expected, printed in pairs:
            with self.subTest(expected=expected, printed=printed):
                sample = problem_sample(expected, {"stdio_exact_output": True})
                verdict, details = check_correctness(
                    sample, f"print({printed!r})", timeout=5, debug=False
                )
                self.assertEqual(verdict, [-2])
                self.assertEqual(details["error_code"], -2)

    def test_exact_mode_preserves_outer_whitespace_handling(self):
        sample = problem_sample("0011\n5.00\n", {"stdio_exact_output": True})
        verdict, _ = check_correctness(
            sample, "print('  0011  \\n 5.00  ')", timeout=5, debug=False
        )
        self.assertEqual(verdict, [True])

    def test_unmarked_and_disabled_tasks_keep_decimal_comparison(self):
        for metadata in ({}, {"stdio_exact_output": False}):
            with self.subTest(metadata=metadata):
                sample = problem_sample("5.00", metadata)
                verdict, _ = check_correctness(
                    sample, "print('5')", timeout=5, debug=False
                )
                self.assertEqual(verdict, [True])

    def test_call_based_comparison_is_unaffected(self):
        sample = problem_sample(
            "5.0",
            {"stdio_exact_output": True, "func_name": "solve"},
            test_type="functional",
            input_text="5",
        )
        verdict, _ = check_correctness(
            sample, "def solve(x): return x", timeout=5, debug=False
        )
        self.assertEqual(verdict, [True])
