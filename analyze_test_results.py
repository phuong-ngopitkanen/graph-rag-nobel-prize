import json
from pathlib import Path
from typing import Any, Dict, List

RESULTS_PATH = Path("test_results.json")
SUMMARY_MD_PATH = Path("summary_report.md")
SUMMARY_JSON_PATH = Path("analysis_summary.json")


def _load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return json.loads(path.read_text())


def _compute_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    explain_success = sum(1 for r in results if r.get("explain_success"))
    first_pass = sum(
        1
        for r in results
        if r.get("explain_success")
        and (r.get("attempts") or [""])[0] == r.get("final_query")
    )
    refinement_needed = explain_success - first_pass
    explain_failures = total - explain_success
    execution_failures = sum(1 for r in results if not r.get("execution_success"))
    postprocessor_pass = sum(
        1
        for r in results
        if r.get("postprocessor_checks", {}).get("lowercase_ok")
        and r.get("postprocessor_checks", {}).get("return_projection_ok")
    )

    return {
        "total": total,
        "explain_success": explain_success,
        "first_pass_success": first_pass,
        "refinement_needed": refinement_needed,
        "explain_failures": explain_failures,
        "execution_failures": execution_failures,
        "postprocessor_pass": postprocessor_pass,
    }


def _collect_examples(results: List[Dict[str, Any]], predicate) -> List[Dict[str, Any]]:
    return [r for r in results if predicate(r)]


def _short_error(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else f"{text[:limit]}..."


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = _compute_stats(results)

    explain_failures = _collect_examples(results, lambda r: not r.get("explain_success"))
    post_lower_failures = _collect_examples(
        results, lambda r: not r.get("postprocessor_checks", {}).get("lowercase_ok")
    )
    return_raw_failures = _collect_examples(
        results, lambda r: not r.get("postprocessor_checks", {}).get("return_projection_ok")
    )
    execution_failures = _collect_examples(results, lambda r: not r.get("execution_success"))
    single_try_success = _collect_examples(
        results,
        lambda r: r.get("explain_success") and (r.get("attempts") or [""])[0] == r.get("final_query"),
    )
    repaired_success = _collect_examples(
        results,
        lambda r: r.get("explain_success")
        and len(r.get("attempts", [])) > 1
        and r.get("final_query") != (r.get("attempts") or [""])[0],
    )

    summary = {
        "stats": stats,
        "failing_explain": [
            {"question": r["question"], "error": _short_error(r.get("errors", {}).get("explain_error", ""))}
            for r in explain_failures
        ],
        "postprocessor_lowercase_issues": [r["question"] for r in post_lower_failures],
        "postprocessor_return_issues": [r["question"] for r in return_raw_failures],
        "execution_failures": [
            {"question": r["question"], "error": _short_error(r.get("errors", {}).get("execution_error", ""))}
            for r in execution_failures
        ],
        "single_attempt_success": [r["question"] for r in single_try_success],
        "repaired_success": [r["question"] for r in repaired_success],
    }

    return summary


def _write_reports(summary: Dict[str, Any]) -> None:
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2))

    stats = summary["stats"]
    lines = [
        "# GraphRAG Few-shot Test Summary",
        "",
        f"- Total examples: {stats['total']}",
        f"- EXPLAIN success: {stats['explain_success']} (first-pass: {stats['first_pass_success']}, refinements: {stats['refinement_needed']})",
        f"- EXPLAIN failures after retries: {stats['explain_failures']}",
        f"- Execution failures: {stats['execution_failures']}",
        f"- Post-processor fully passing: {stats['postprocessor_pass']}",
        "",
        "## Problematic Examples",
        "",
        "### EXPLAIN failures",
    ]
    if summary["failing_explain"]:
        lines.extend([f"- {item['question']} — {item['error']}" for item in summary["failing_explain"]])
    else:
        lines.append("- None")

    lines.extend(["", "### Post-processor issues (lowercase)", ""])
    if summary["postprocessor_lowercase_issues"]:
        lines.extend([f"- {q}" for q in summary["postprocessor_lowercase_issues"]])
    else:
        lines.append("- None")

    lines.extend(["", "### Post-processor issues (RETURN projection)", ""])
    if summary["postprocessor_return_issues"]:
        lines.extend([f"- {q}" for q in summary["postprocessor_return_issues"]])
    else:
        lines.append("- None")

    lines.extend(["", "### Execution failures", ""])
    if summary["execution_failures"]:
        lines.extend([f"- {item['question']} — {item['error']}" for item in summary["execution_failures"]])
    else:
        lines.append("- None")

    lines.extend(["", "## Strengths", ""])
    if summary["single_attempt_success"]:
        lines.append(f"- Single-attempt successes: {len(summary['single_attempt_success'])}")
    else:
        lines.append("- Single-attempt successes: 0")
    if summary["repaired_success"]:
        lines.append(f"- Successful repairs after EXPLAIN feedback: {len(summary['repaired_success'])}")
    else:
        lines.append("- Successful repairs after EXPLAIN feedback: 0")

    SUMMARY_MD_PATH.write_text("\n".join(lines))


def main() -> None:
    results = _load_results(RESULTS_PATH)
    summary = _build_summary(results)
    _write_reports(summary)

    stats = summary["stats"]
    print(f"Total examples: {stats['total']}")
    print(f"EXPLAIN success: {stats['explain_success']} (first-pass: {stats['first_pass_success']}, refinements: {stats['refinement_needed']})")
    print(f"EXPLAIN failures after retries: {stats['explain_failures']}")
    print(f"Execution failures: {stats['execution_failures']}")
    print(f"Post-processor fully passing: {stats['postprocessor_pass']}")

    if summary["failing_explain"]:
        print("\nExamples failing EXPLAIN:")
        for item in summary["failing_explain"]:
            print(f"- {item['question']} — {item['error']}")

    if summary["postprocessor_lowercase_issues"]:
        print("\nRETURN/string normalization issues (lowercase):")
        for question in summary["postprocessor_lowercase_issues"]:
            print(f"- {question}")

    if summary["postprocessor_return_issues"]:
        print("\nRETURN projection issues:")
        for question in summary["postprocessor_return_issues"]:
            print(f"- {question}")

    if summary["execution_failures"]:
        print("\nExecution failures:")
        for item in summary["execution_failures"]:
            print(f"- {item['question']} — {item['error']}")


if __name__ == "__main__":
    main()
