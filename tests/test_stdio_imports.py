import signal
import unittest

from lcb_runner.evaluation.testing_util import grade_stdio, timeout_handler


class StdioImportsTests(unittest.TestCase):
    def setUp(self):
        self.previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
        self.addCleanup(signal.signal, signal.SIGALRM, self.previous_handler)

    def check_program(self, code, expected=True):
        grades, _ = grade_stdio(code, ["2 3", "4 5", "7 8"], ["6", "20", "56"], 3)
        self.assertEqual(grades, [True] * 3 if expected else [-2])

    def test_stdout_alias_is_bound_during_capture_for_every_case(self):
        self.check_program(
            "from sys import stdout\na,b=map(int,input().split())\n"
            "stdout.write(str(a*b)+'\\n')"
        )

    def test_renamed_alias_and_nested_function(self):
        self.check_program(
            "from sys import stdout as output\n"
            "def answer(value):\n    output.write(str(value)+'\\n')\n"
            "a,b=map(int,input().split())\nanswer(a*b)"
        )

    def test_stdin_alias_is_bound_to_each_cases_input(self):
        self.check_program(
            "from sys import stdin, stdout\n"
            "a,b=map(int,stdin.readline().split())\nstdout.write(str(a*b))"
        )

    def test_module_attribute_control(self):
        self.check_program(
            "import sys\na,b=map(int,input().split())\nsys.stdout.write(str(a*b))"
        )

    def test_wrong_answer_is_still_rejected(self):
        self.check_program(
            "from sys import stdout\na,b=map(int,input().split())\nstdout.write(str(a+b))",
            expected=False,
        )

    def test_wildcard_import_remains_legal(self):
        self.check_program(
            "from math import *\na,b=map(int,input().split())\nprint(prod([a,b]))"
        )

    def test_import_order_preserves_local_rebinding(self):
        self.check_program(
            "a,b=map(int,input().split())\nsys = a*b\n"
            "def result():\n    return sys\nprint(result())"
        )

    def test_imports_stay_visible_to_global_declarations(self):
        self.check_program(
            "import math as maths\ndef solve():\n    global maths\n"
            "    a,b=map(int,input().split())\n    print(maths.prod([a,b]))\nsolve()"
        )

    def test_wildcard_import_order_preserves_shadowing(self):
        self.check_program(
            "import math as pi\nfrom math import *\n"
            "a,b=map(int,input().split())\nprint(int(pi*0)+a*b)"
        )

    def test_module_globals_still_contain_import_aliases(self):
        self.check_program(
            "import math as maths\na,b=map(int,input().split())\n"
            "print(globals()['maths'].prod([a,b]))"
        )


if __name__ == "__main__":
    unittest.main()
