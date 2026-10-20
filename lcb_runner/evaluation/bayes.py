"""Binary Bayes@N summaries (https://arxiv.org/abs/2510.04265)."""

import math
from numbers import Real


def bayes_at_n(results: list[dict]) -> dict:
    """Summarize average success probability with independent Beta(1, 1) priors.

    Each record is a question with an ID and repeated binary `graded_list`.
    All questions must have the same positive number of independent generations.
    Returns the posterior mean and SD on the [0, 1] score scale, plus cohort size.
    """
    if not results:
        raise ValueError("Bayes@N needs at least one question")
    identifiers = set()
    counts = []
    trial_count = None
    for result in results:
        identifier = result.get("question_id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Bayes@N needs unique, nonempty question_id strings")
        identifiers.add(identifier)
        grades = result.get("graded_list")
        if not isinstance(grades, list) or not grades:
            raise ValueError("Bayes@N needs a nonempty graded_list for every question")
        if any(not isinstance(g, Real) or g not in (0, 1) for g in grades):
            raise ValueError("Bayes@N needs binary grades (0/1 or false/true)")
        if trial_count is None:
            trial_count = len(grades)
        elif len(grades) != trial_count:
            raise ValueError("Bayes@N needs the same sample count for every question")
        counts.append(sum(grades))

    total = trial_count + 2
    means = [(c + 1) / total for c in counts]
    variances = [((c + 1) / total) * ((trial_count - c + 1) / total)
                 / (total + 1) for c in counts]
    return {
        "mean": math.fsum(means) / len(results),
        "std": math.sqrt(math.fsum(variances)) / len(results),
        "num_questions": len(results),
        "samples_per_question": trial_count,
    }
