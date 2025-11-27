
import importlib.util
import inspect
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

GRAPH_RAG_PATH = Path(__file__).parent / "graph_rag.py"
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)


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

    required_keys = ["GraphRAG", "KuzuDatabaseManager", "FEW_SHOT_EXAMPLES"]
    missing = [k for k in required_keys if k not in env]
    if missing:
        raise RuntimeError(f"Missing components after load: {missing}")
    return env


def load_test_queries() -> Dict[str, List[str]]:
    
    # Query set 1: Repeated queries that contains exact same texts
    exact_repeats = [
        "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        "How many laureates were born in Germany?",
        "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        "List all female laureates who won prizes in Chemistry.",
        "How many laureates were born in Germany?",
        "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        "List all female laureates who won prizes in Chemistry.",
        "How many laureates were born in Germany?",
        "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        "What is the total prize amount awarded in Physics?",
    ]
    
    # Query set 2: Queries that are semantically related but textually different
    similar_queries = [
        "Which scholars won prizes in Physics?",
        "How many scholars won prizes in Physics?",
        "List all scholars who won a Physics prize.",
        "Which institutions have affiliated Physics laureates?",
        "How many Physics prizes were awarded from 2000 onwards?",
        "Which Physics laureates were born in Europe?",
        "Which scholars won prizes in Physics?",
        "How many scholars won prizes in Physics?",
        "Which Physics laureates were affiliated with institutions in Japan?",
        "Which Physics laureates were born in cities located in Asia?",
    ]
    
    # Query set 3: Queries that are unique with no repetition
    diverse_queries = [
        "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
        "How many laureates were born in Germany?",
        "List all female laureates who won prizes in Chemistry.",
        "What is the total prize amount awarded in Physics?",
        "Which Economics laureates were affiliated with Harvard University?",
        "How many laureates died in the United States?",
        "Which continents have Chemistry laureates by birth?",
        "For each prize category, how many distinct laureates are there?",
        "Which institutions have hosted laureates from more than one prize category?",
        "Which cities host institutions for both Physics and Chemistry laureates?",
    ]
    
    # Query set 4: A mix of new queries, repeats and related questions 
    mixed_pattern = [
        "Which scholars won prizes in Physics?",
        "How many scholars won prizes in Chemistry?",
        "Which scholars won prizes in Physics?",  
        "List all scholars who won a Medicine prize.",
        "How many scholars won prizes in Chemistry?",  
        "Which continents have Chemistry laureates by birth?",
        "Which scholars won prizes in Physics?",  
        "Which institutions have affiliated Physics laureates?",
        "List all scholars who won a Medicine prize.",  
        "How many prizes were awarded in Economics after 1980?",
    ]
    
    # Query set 5: Computionally expensive queries to stress test the GraphRAG system
    complex_queries = [
        "Which scholars were born in the United States but affiliated with the University of Cambridge?",
        "For each prize category, how many distinct laureates are there?",
        "Which countries have produced laureates in both Physics and Chemistry by birth?",
        "Which laureates were affiliated with institutions on a different continent than their birth continent?",
        "For each continent, how many distinct laureates have affiliated institutions there?",
        "Which scholars were born in the United States but affiliated with the University of Cambridge?", 
        "Which countries have produced laureates who won prizes in both Medicine and Physics?",
        "For each birth continent, how many laureates later worked at institutions on a different continent?",
        "Which scholars have affiliations in at least three different countries?",
        "Which laureates were born in cities that host institutions they were also affiliated with?",
    ]
    
    return {
        "exact_repeats": exact_repeats,
        "similar_queries": similar_queries,
        "diverse_queries": diverse_queries,
        "mixed_pattern": mixed_pattern,
        "complex_queries": complex_queries,
    }


def run_test_scenario(
    GraphRAG,
    KuzuDatabaseManager,
    cache_size: int,
    query_set_name: str,
    queries: List[str],
    db_path: str = "nobel.kuzu"
) -> Dict[str, Any]:
        
    print(f"\n{'='*80}")
    print(f"Testing: Cache size = {cache_size}, Query set = {query_set_name}")
    print(f"{'='*80}\n")
    
    db_manager = KuzuDatabaseManager(db_path)
    schema = str(db_manager.get_schema_dict)
    
    rag = GraphRAG(use_few_shot=True, k_examples=3, cache_size=cache_size)
    
    results = {
        "cache_size": cache_size,
        "query_set": query_set_name,
        "total_queries": len(queries),
        "queries": [],
        "summary": {}
    }
    
    total_time = 0
    total_cache_hits = 0
    total_cache_misses = 0
    
    for idx, question in enumerate(queries, 1):
        print(f"Query {idx}/{len(queries)}: {question[:60]}...")
        
        initial_hits = rag.cache.stats.hits
        initial_misses = rag.cache.stats.misses
        
        start_time = time.time()
        
        try:
            response = rag(
                db_manager=db_manager,
                question=question,
                input_schema=schema
            )
            
            query_time = time.time() - start_time
            
            cache_hit = (rag.cache.stats.hits > initial_hits)
            
            query_result = {
                "query_no": idx,
                "question": question,
                "cypher_query": response.get("query", ""),
                "execution_time": query_time,
                "cache_hit": cache_hit,
                "success": True
            }
            
            print(f" Completed in {query_time:.3f}s | Cache: {'HIT' if cache_hit else 'MISS'}")
            
        except Exception as e:
            query_time = time.time() - start_time
            query_result = {
                "query_no": idx,
                "question": question,
                "execution_time": query_time,
                "cache_hit": False,
                "success": False,
                "error": str(e)
            }
            print(f" Failed in {query_time:.3f}s: {str(e)[:50]}")
        
        results["queries"].append(query_result)
        total_time += query_time
        
        if query_result.get("cache_hit"):
            total_cache_hits += 1
        else:
            total_cache_misses += 1
    
    cache_stats = rag.get_cache_stats()
    
    results["summary"] = {
        "total_execution_time": total_time,
        "average_query_time": total_time / len(queries),
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "cache_hit_rate": (total_cache_hits / len(queries)) * 100 if len(queries) > 0 else 0,
        "final_cache_stats": cache_stats
    }
    
    print(f"\n{'='*60}")
    print(f"SUMMARY - {query_set_name} (cache_size={cache_size})")
    print(f"{'='*60}")
    print(f"Total queries: {len(queries)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time/len(queries):.3f}s")
    print(f"Cache hits: {total_cache_hits}")
    print(f"Cache misses: {total_cache_misses}")
    print(f"Hit rate: {results['summary']['cache_hit_rate']:.1f}%")
    print(f"{'='*60}\n")
    
    return results


def run_full_test_suite(
    cache_sizes: List[int] = None,
    db_path: str = "nobel.kuzu"
) -> List[Dict[str, Any]]:    
    if cache_sizes is None:
        cache_sizes = [0, 3, 10, 50, 128]
    
    print("\n" + "#"*80)
    print("Loading GraphRAG components from marimo notebook...")
    print("#"*80 + "\n")
    
    components = _load_graph_rag_components()
    GraphRAG = components["GraphRAG"]
    KuzuDatabaseManager = components["KuzuDatabaseManager"]
    
    print("Components loaded successfully\n")
    
    query_sets = load_test_queries()
    all_results = []
    
    print(f"\n{'#'*80}")
    print(f"# GraphRAG cache performance test suite")
    print(f"# Database: {db_path}")
    print(f"# Cache Sizes: {cache_sizes}")
    print(f"# Query Sets: {list(query_sets.keys())}")
    print(f"{'#'*80}\n")
    
    for cache_size in cache_sizes:
        for query_set_name, queries in query_sets.items():
            result = run_test_scenario(
                GraphRAG,
                KuzuDatabaseManager,
                cache_size,
                query_set_name,
                queries,
                db_path
            )
            all_results.append(result)
            
            time.sleep(1)
    
    return all_results


def save_results(results: List[Dict[str, Any]], filename: str = None) -> Path:    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cache_performance_test_{timestamp}.json"
    
    output_path = OUTPUT_DIR / filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to: {output_path}")
    return output_path


def generate_summary_report(results: List[Dict[str, Any]]) -> Path:    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"summary_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# GraphRAG cache performance test report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        by_cache_size = {}
        for result in results:
            cache_size = result["cache_size"]
            if cache_size not in by_cache_size:
                by_cache_size[cache_size] = []
            by_cache_size[cache_size].append(result)
        
        f.write("## Summary by cache size\n\n")
        
        for cache_size in sorted(by_cache_size.keys()):
            f.write(f"### Cache Size: {cache_size}\n\n")
            f.write("| Query Set | Total Time (s) | Avg Time (s) | Cache Hits | Hit Rate (%) |\n")
            f.write("|-----------|----------------|--------------|------------|-------------|\n")
            
            for result in by_cache_size[cache_size]:
                summary = result["summary"]
                f.write(f"| {result['query_set']} | "
                       f"{summary['total_execution_time']:.2f} | "
                       f"{summary['average_query_time']:.3f} | "
                       f"{summary['cache_hits']}/{result['total_queries']} | "
                       f"{summary['cache_hit_rate']:.1f} |\n")
            
            f.write("\n")
        
        f.write("## Key findings\n\n")
        
        f.write("### Optimal cache size by query pattern\n\n")
        
        query_sets = set(r["query_set"] for r in results)
        for query_set in query_sets:
            relevant = [r for r in results if r["query_set"] == query_set]
            best = min(relevant, key=lambda x: x["summary"]["average_query_time"])
            
            f.write(f"- **{query_set}**: Cache size {best['cache_size']} "
                   f"({best['summary']['average_query_time']:.3f}s avg, "
                   f"{best['summary']['cache_hit_rate']:.1f}% hit rate)\n")
        
        f.write("\n### Cache hit rate analysis\n\n")
        
        for query_set in query_sets:
            relevant = [r for r in results if r["query_set"] == query_set]
            f.write(f"\n**{query_set}**:\n")
            for r in sorted(relevant, key=lambda x: x["cache_size"]):
                hit_rate = r["summary"]["cache_hit_rate"]
                f.write(f"- Cache {r['cache_size']}: {hit_rate:.1f}% hit rate\n")
    
    print(f"Summary report saved to: {report_path}")
    return report_path


def main():    
    CACHE_SIZES = [0, 3, 10, 50, 128]
    DB_PATH = "nobel.kuzu"
    
    print("\n" + "="*80)
    print("GraphRAG cache performance test")
    print("="*80)
    
    print("\nStarting tests...")
    results = run_full_test_suite(cache_sizes=CACHE_SIZES, db_path=DB_PATH)
    
    json_path = save_results(results)
    report_path = generate_summary_report(results)
    
    print(f"\n{'='*80}")
    print(f"Test suite completed")
    print(f"{'='*80}")
    print(f"Results saved to:")
    print(f"  - JSON: {json_path}")
    print(f"  - Report: {report_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()