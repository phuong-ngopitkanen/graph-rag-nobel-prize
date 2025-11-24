import importlib.util
import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

RESULTS_PATH = Path("test_results.json")
GRAPH_RAG_PATH = Path(__file__).parent / "graph_rag.py"


class _FakeApp:
    def __init__(self, *args, **kwargs) -> None:
        self.cells: List[Any] = []

    def cell(self, *args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            fn = args[0]
            self.cells.append(fn)
            return fn

        def decorator(fn):
            self.cells.append(fn)
            return fn

        return decorator

    def run(self, *args, **kwargs) -> None:
        return


def _capture_locals(func, kwargs: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    """Run a cell function and capture its locals at return time."""
    captured: Dict[str, Any] = {}
    previous_profile = sys.getprofile()

    def tracer(frame, event, arg):
        if event == "return" and frame.f_code is func.__code__:
            captured.update(frame.f_locals)
        return tracer

    sys.setprofile(tracer)
    try:
        result = func(**kwargs)
    finally:
        sys.setprofile(previous_profile)
    return result, captured


def _load_graph_rag_components() -> Dict[str, Any]:
    original_marimo = sys.modules.get("marimo")
    marimo_stub = types.SimpleNamespace(App=_FakeApp)
    sys.modules["marimo"] = marimo_stub

    spec = importlib.util.spec_from_file_location("graph_rag_module", GRAPH_RAG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load graph_rag.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["graph_rag_module"] = module
    spec.loader.exec_module(module)

    cells: List[Any] = getattr(module.app, "cells", [])
    env: Dict[str, Any] = {}

    def find_cell(needle: str):
        for cell in cells:
            source = inspect.getsource(cell)
            if needle in source:
                return cell
        raise RuntimeError(f"Cell containing '{needle}' not found")

    def run_cell(cell, capture: bool = False):
        sig = inspect.signature(cell)
        kwargs = {name: env[name] for name in sig.parameters}
        if capture:
            return _capture_locals(cell, kwargs)
        return cell(**kwargs), {}

    try:
        import_cell = find_cell("BAMLAdapter")
        result, _ = run_cell(import_cell, capture=True)
        (
            AnyType,
            BAMLAdapter,
            BaseModel,
            Field,
            dspy_mod,
            json_mod,
            kuzu_mod,
            mo_mod,
            np_mod,
            os_mod,
            _re_mod,
        ) = result
        env.update(
            {
                "Any": AnyType,
                "BAMLAdapter": BAMLAdapter,
                "BaseModel": BaseModel,
                "Field": Field,
                "dspy": dspy_mod,
                "json": json_mod,
                "kuzu": kuzu_mod,
                "mo": mo_mod,
                "np": np_mod,
                "os": os_mod,
            }
        )
        module.__dict__.update(env)

        schema_cell = find_cell("class Query")
        _, locals_out = run_cell(schema_cell, capture=True)
        env.update(locals_out)
        module.__dict__.update(locals_out)

        signatures_cell = find_cell("class PruneSchema")
        _, locals_out = run_cell(signatures_cell, capture=True)
        env.update(locals_out)
        module.__dict__.update(locals_out)

        config_cell = find_cell("GEMINI_API_KEY")
        run_cell(config_cell, capture=False)

        kuzu_cell = find_cell("class KuzuDatabaseManager")
        result, locals_out = run_cell(kuzu_cell, capture=True)
        if isinstance(result, tuple):
            env["KuzuDatabaseManager"] = result[0]
        env.update(locals_out)
        module.__dict__.update(env)

        few_shot_data_cell = find_cell("few_shot_examples.json")
        result, locals_out = run_cell(few_shot_data_cell, capture=True)
        env["FEW_SHOT_EXAMPLES"] = result[0] if isinstance(result, tuple) else result
        env.update(locals_out)

        retriever_cell = find_cell("FewShotRetriever")
        result, locals_out = run_cell(retriever_cell, capture=True)
        env["FewShotRetriever"] = result[0] if isinstance(result, tuple) else result
        env.update(locals_out)

        graphrag_cell = find_cell("class GraphRAG")
        result, locals_out = run_cell(graphrag_cell, capture=True)
        if isinstance(result, tuple):
            env["GraphRAG"] = result[0]
            env["run_graph_rag"] = result[1] if len(result) > 1 else None
        env.update(locals_out)
        module.__dict__.update(env)
    finally:
        if original_marimo is not None:
            sys.modules["marimo"] = original_marimo
        else:
            sys.modules.pop("marimo", None)

    required_keys = ["GraphRAG", "FewShotRetriever", "FEW_SHOT_EXAMPLES", "KuzuDatabaseManager", "np", "dspy", "kuzu"]
    missing = [k for k in required_keys if k not in env]
    if missing:
        raise RuntimeError(f"Missing components after load: {missing}")
    return env


def _compute_retriever_similarities(retriever, question: str, np_mod) -> List[Dict[str, Any]]:
    question_embedding = retriever.model.encode(
        [question],
        convert_to_tensor=False,
        show_progress_bar=False,
    )[0]
    scores = []
    for ex, ex_emb in zip(retriever.examples, retriever.example_embeddings):
        sim = float(np_mod.dot(question_embedding, ex_emb) / (np_mod.linalg.norm(question_embedding) * np_mod.linalg.norm(ex_emb)))
        scores.append({"question": ex["question"], "similarity": sim})
    scores.sort(key=lambda item: item["similarity"], reverse=True)
    return scores[: retriever.k]


def _has_bare_return(query: str) -> bool:
    match = re.search(r"(?i)\breturn\b", query)
    if not match:
        return False
    body_and_tail = query[match.end() :]
    tail_start = len(body_and_tail)
    for pattern in (r"\bORDER\s+BY\b", r"\bLIMIT\b", r"\bSKIP\b"):
        found = re.search(pattern, body_and_tail, flags=re.IGNORECASE)
        if found and found.start() < tail_start:
            tail_start = found.start()
    return_body = body_and_tail[:tail_start].strip()
    if return_body.upper().startswith("DISTINCT "):
        return_body = return_body[len("DISTINCT ") :].strip()
    items = []
    current = []
    depth = 0
    in_single = False
    in_double = False
    for ch in return_body:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
        if ch == "," and depth == 0 and not in_single and not in_double:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    remainder = "".join(current).strip()
    if remainder:
        items.append(remainder)
    for item in items:
        alias_match = re.match(r"(?is)(.+?)\s+AS\s+([A-Za-z_]\w*)$", item)
        expr = alias_match.group(1).strip() if alias_match else item
        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            return True
    return False


def _lowercase_normalized(query: str) -> bool:
    text_properties = {
        "birthDate",
        "deathDate",
        "fullName",
        "gender",
        "knownName",
        "name",
        "category",
        "motivation",
        "scholar_type",
        "state",
        "portion",
        "dateAwarded",
        "prize_id",
    }
    property_pattern = "|".join(re.escape(p) for p in sorted(text_properties))
    comparison_pattern = re.compile(
        rf"(?is)(toLower\s*\(\s*)?([A-Za-z_]\w*\.({property_pattern}))\s*(\))?\s*(=|<>|!=|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*(toLower\s*\(\s*)?('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*\)?"
    )
    violations = 0
    for match in comparison_pattern.finditer(query):
        prop_lower = bool(match.group(1))
        literal_lower = bool(match.group(6))
        if not (prop_lower and literal_lower):
            violations += 1
    return violations == 0


def _process_example(
    example: Dict[str, Any],
    rag,
    retriever,
    db_manager,
    schema_text: str,
    np_mod,
) -> Dict[str, Any]:
    question = example["question"]
    expected_query = example.get("cypher_query", "")
    attempts: List[str] = []
    explain_error = ""
    execution_error = ""
    final_query = ""
    explain_success = False
    execution_success = False
    previous_query = ""
    error_message = ""

    max_attempts = 3
    initial_query = ""
    for _ in range(max_attempts):
        try:
            candidate_query = rag.get_cypher_query(
                question=question,
                input_schema=schema_text,
                previous_query=previous_query or None,
                error_message=error_message or None,
            )
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            attempts.append("")
            break

        if not initial_query:
            initial_query = candidate_query
        attempts.append(candidate_query)

        try:
            db_manager.conn.execute(f"EXPLAIN {candidate_query}")
            explain_success = True
            final_query = candidate_query
            break
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            explain_error = error_message
            previous_query = candidate_query
            continue

    if not final_query:
        final_query = previous_query

    if final_query:
        final_query = rag.postprocess_cypher(final_query)
        try:
            db_manager.conn.execute(final_query)
            execution_success = True
        except Exception as exc:  # noqa: BLE001
            execution_error = str(exc)

    postprocessor_checks = {
        "lowercase_ok": _lowercase_normalized(final_query) if final_query else False,
        "return_projection_ok": not _has_bare_return(final_query) if final_query else False,
    }

    retriever_similarities = _compute_retriever_similarities(retriever, question, np_mod) if retriever else []

    return {
        "question": question,
        "expected_query": expected_query,
        "initial_query": initial_query,
        "attempts": attempts,
        "final_query": final_query,
        "explain_success": explain_success,
        "execution_success": execution_success,
        "retriever_similarities": retriever_similarities,
        "postprocessor_checks": postprocessor_checks,
        "errors": {
            "explain_error": explain_error,
            "execution_error": execution_error,
        },
    }


def main() -> None:
    components = _load_graph_rag_components()
    GraphRAG = components["GraphRAG"]
    FewShotRetriever = components["FewShotRetriever"]
    FEW_SHOT_EXAMPLES = components["FEW_SHOT_EXAMPLES"]
    KuzuDatabaseManager = components["KuzuDatabaseManager"]
    np_mod = components["np"]

    db_manager = KuzuDatabaseManager("nobel.kuzu")
    schema_text = str(db_manager.get_schema_dict)

    rag = GraphRAG(use_few_shot=True, k_examples=3)
    retriever = rag.few_shot_retriever
    if retriever is None:
        retriever = FewShotRetriever(examples=FEW_SHOT_EXAMPLES, k=3)

    results: List[Dict[str, Any]] = []
    for example in FEW_SHOT_EXAMPLES:
        result = _process_example(example, rag, retriever, db_manager, schema_text, np_mod)
        results.append(result)

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved test results for {len(results)} examples to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
