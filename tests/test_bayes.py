import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lcb_runner.evaluation.bayes import bayes_at_n
from lcb_runner.evaluation.compute_scores import compute_scores
from lcb_runner.utils.scenarios import Scenario


def records(rows):
    return [{"question_id": str(i), "graded_list": row} for i, row in enumerate(rows)]


@pytest.mark.parametrize("rows,mean,variance", [
    ([[0, 0], [1, 1]], 0.5, 3 / 160),
    ([[0, 1], [0, 1]], 0.5, 1 / 40),
    ([[0]], 1 / 3, 1 / 18),
    ([[1]], 2 / 3, 1 / 18),
])
def test_exact_moments(rows, mean, variance):
    result = bayes_at_n(records(rows))
    assert result["mean"] == pytest.approx(mean)
    assert result["std"] == pytest.approx(math.sqrt(variance))
    assert result["num_questions"] == len(rows)
    assert result["samples_per_question"] == len(rows[0])


def test_combined_resume_evidence_and_permutations():
    old = records([[0, 0]])
    new = [{"question_id": "new", "graded_list": [0, 1]}]
    result = bayes_at_n(old + new)
    assert result["mean"] == pytest.approx(3 / 8)
    assert result["std"] == pytest.approx(math.sqrt(7 / 320))
    assert result == bayes_at_n(new + old)
    assert result == bayes_at_n([dict(r, graded_list=r["graded_list"][::-1]) for r in old + new])
    assert result["std"] != pytest.approx((bayes_at_n(old)["std"] + bayes_at_n(new)["std"]) / 2)


@pytest.mark.parametrize("bad", [None, [], [0, 1, 0], [None, 1], ["1", 0],
                                 [0.5, 1], [float("nan"), 0], [-1, 1], [2, 0]])
def test_invalid_rows(bad):
    data = records([[0, 0], [1, 1]])
    data[1]["graded_list"] = bad
    with pytest.raises(ValueError, match="Bayes@N"):
        bayes_at_n(data)


@pytest.mark.parametrize("bad", [None, "", "0", 1])
def test_missing_or_duplicate_ids(bad):
    data = records([[0, 0], [1, 1]])
    data[1]["question_id"] = bad
    with pytest.raises(ValueError, match="question_id"):
        bayes_at_n(data)


def test_empty_corpus():
    with pytest.raises(ValueError, match="at least one"):
        bayes_at_n([])


def input_file(tmp_path):
    data = records([[0, 0], [1, 1], [0, 1], [1, 1]])
    for i, row in enumerate(data):
        row.update(contest_date="2024-01-02T00:00:00", platform="leetcode",
                   difficulty=["easy", "medium", "hard", "hard"][i],
                   **{"pass@1": sum(row["graded_list"]) / 2})
    data[3]["contest_date"] = "2023-01-01T00:00:00"
    data[2]["platform"] = "atcoder"
    path = tmp_path / "eval_all.json"
    path.write_text(json.dumps(data))
    return path


def cli(path, *extra):
    return subprocess.run([sys.executable, "-m", "lcb_runner.evaluation.compute_scores",
                           "--eval_all_file", str(path), *extra],
                          capture_output=True, text=True)


def test_documented_cli_filters_and_json(tmp_path):
    path = input_file(tmp_path)
    before = path.read_bytes()
    output = tmp_path / "posterior.json"
    run = cli(path, "--bayes_at_n", "--bayes_output_file", str(output),
              "--start_date", "2024-01-01", "--end_date", "2024-12-31",
              "--platform", "leetcode")
    assert run.returncode == 0, run.stderr
    assert "Bayes@N (overall)" in run.stdout
    assert "Pass@1" in run.stdout
    result = json.loads(output.read_text())
    assert set(result) == {"overall", "easy", "medium"}
    assert result["overall"] == pytest.approx({"mean": 0.5, "std": math.sqrt(3 / 160),
                                             "num_questions": 2, "samples_per_question": 2})
    assert result["easy"]["std"] == pytest.approx(math.sqrt(3 / 80))
    assert result["medium"]["mean"] == 0.75
    assert path.read_bytes() == before


def test_default_cli_has_no_posterior(tmp_path):
    run = cli(input_file(tmp_path))
    assert run.returncode == 0, run.stderr
    assert "Bayes@N" not in run.stdout
    assert "Pass@1" in run.stdout


@pytest.mark.parametrize("extra", [
    ["--bayes_output_file", "unused.json"],
    ["--bayes_at_n", "--scenario", "selfrepair"],
    ["--bayes_at_n", "--start_date", "2030-01-01"],
])
def test_cli_invalid_scope(tmp_path, extra):
    run = cli(input_file(tmp_path), *extra)
    assert run.returncode != 0
    assert "Bayes@N" in run.stderr or "requires --bayes_at_n" in run.stderr


def test_refuse_input_overwrite(tmp_path):
    path = input_file(tmp_path)
    before = path.read_bytes()
    run = cli(path, "--bayes_at_n", "--bayes_output_file", str(path))
    assert run.returncode != 0
    assert path.read_bytes() == before


def test_python_public_interface_uses_filtered_records(tmp_path, capsys):
    path = input_file(tmp_path)
    compute_scores(SimpleNamespace(
        eval_all_file=str(path), start_date="2024-01-01", end_date=None,
        platform="leetcode", scenario=Scenario.codegeneration, bayes_at_n=True,
    ))
    assert "Bayes@N (overall): mean=0.500000, std=0.136931" in capsys.readouterr().out
