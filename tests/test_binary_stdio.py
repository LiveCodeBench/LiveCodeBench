import signal
import sys
import unittest

from lcb_runner.evaluation.testing_util import (
    Capturing,
    MockBuffer,
    MockStdinWithBuffer,
    call_method,
    grade_stdio,
    timeout_handler,
)


class BinaryStdioTests(unittest.TestCase):
    def test_mock_buffer_advances_for_read_and_readline(self):
        stream = MockBuffer("ab\ncd\n")
        self.assertEqual(stream.read(1), b"a")
        self.assertEqual(stream.readline(), b"b\n")
        self.assertEqual(stream.readline(), b"cd\n")
        self.assertEqual(stream.readline(), b"")

    def test_text_and_binary_reads_share_a_cursor(self):
        stream = MockStdinWithBuffer("first\nsecond\nthird\n")
        self.assertEqual(stream.readline(), "first\n")
        self.assertEqual(stream.buffer.readline(), b"second\n")
        self.assertEqual(stream.readline(), "third\n")
        self.assertEqual(stream.buffer.readline(), b"")

    def test_sized_text_reads_count_characters(self):
        stream = MockStdinWithBuffer("中A\nend")
        self.assertEqual(stream.read(1), "中")
        self.assertEqual(stream.readline(2), "A\n")
        self.assertEqual(stream.read(), "end")

    def test_readlines_and_iteration_use_remaining_input(self):
        stream = MockStdinWithBuffer("a\nb\nc")
        self.assertEqual(stream.readline(), "a\n")
        self.assertEqual(stream.readlines(), ["b\n", "c"])
        self.assertEqual(list(stream), [])

    def test_call_method_mixes_input_methods(self):
        seen = []

        def solution():
            seen.append(input())
            seen.append(sys.stdin.buffer.readline().decode().strip())
            seen.append(sys.stdin.readline().strip())

        call_method(solution, "one\ntwo\nthree\n")
        self.assertEqual(seen, ["one", "two", "three"])

    def test_capture_preserves_text_binary_write_order(self):
        original_stdout = sys.stdout
        with Capturing() as output:
            sys.stdout.write("a")
            sys.stdout.buffer.write(b"b")
            print("c")
            sys.stdout.close()
        self.assertIs(sys.stdout, original_stdout)
        self.assertEqual(output, ["abc\n"])

    def test_stdio_judge_accepts_repeated_binary_reads(self):
        code = (
            "import sys\n"
            "a = int(sys.stdin.buffer.readline())\n"
            "b = int(sys.stdin.buffer.readline())\n"
            "print(a + b)\n"
        )
        signal.signal(signal.SIGALRM, timeout_handler)
        grades, _ = grade_stdio(code, ["1\n2\n"], ["3\n"], timeout=3)
        self.assertEqual(grades, [True])

    def test_stdio_judge_accepts_binary_output(self):
        code = (
            "import sys\n"
            "value = sys.stdin.buffer.read().strip()\n"
            "sys.stdout.buffer.write(value + b'!\\n')\n"
        )
        signal.signal(signal.SIGALRM, timeout_handler)
        grades, _ = grade_stdio(code, ["hello\n"], ["hello!\n"], timeout=3)
        self.assertEqual(grades, [True])


if __name__ == "__main__":
    unittest.main()
