import json
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset

class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)

@dataclass
class GenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)

        self.public_test_cases = json.loads(self.public_test_cases)
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        self.private_test_cases = json.loads(self.private_test_cases)
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)


def load_generation_dataset() -> list[GenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    return [GenerationProblem(**p) for p in dataset]