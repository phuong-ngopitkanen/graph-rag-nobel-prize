import marimo
import os
import json
import re
import time
from performance_data import PerformanceDataExporter

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        rf"""
    # Graph RAG using Text2Cypher

    This is a demo app in marimo that allows you to query the Nobel laureate graph (that's managed in Kuzu) using natural language. A language model takes in the question you enter, translates it to Cypher via a custom Text2Cypher pipeline in Kuzu that's powered by DSPy. The response retrieved from the graph database is then used as context to formulate the answer to the question.

    > \- Powered by Kuzu, DSPy and marimo \-
    """
    )
    return


@app.cell
def _(mo):
    text_ui = mo.ui.text(value="Which scholars won prizes in Physics and were affiliated with University of Cambridge?", full_width=True)
    return (text_ui,)


@app.cell
def _(text_ui):
    text_ui
    return

@app.cell
def _(KuzuDatabaseManager,  GraphRAG):

    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)
    rag_instance = GraphRAG(use_few_shot=True, k_examples=3, cache_size=3)

    return (db_manager, rag_instance)


@app.cell
def _(db_manager, rag_instance, mo, text_ui):
    question = text_ui.value
    schema = str(db_manager.get_schema_dict)
    
    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = rag_instance(
            db_manager=db_manager,
            question=question,
            input_schema=schema
        )
    
    # Be defensive in case result is empty or missing keys
    if not result or "query" not in result or "answer" not in result:
        query = ""
        answer = "I couldn't generate a valid answer from the graph. Please try another question."
    else:
        query = result["query"]
        answer = result["answer"].response
    
    return answer, query


@app.cell
def _(answer, mo, query):
    mo.hstack([mo.md(f"""### Query\n```{query}```"""), mo.md(f"""### Answer\n{answer}""")])
    return


@app.cell
def _(dspy):
    class PruneSchema(dspy.Signature):
        """
        Understand the given labelled property graph schema and the given user question. Your task
        is to return ONLY the subset of the schema (node labels, edge labels and properties) that is
        relevant to the question.
            - The schema is a list of nodes and edges in a property graph.
            - The nodes are the entities in the graph.
            - The edges are the relationships between the nodes.
            - Properties of nodes and edges are their attributes, which helps answer the question.
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        pruned_schema: dict = dspy.OutputField()


    class Text2Cypher(dspy.Signature):
        """
        You are an expert Cypher engineer in a GraphRAG system. Your job is to GENERATE and REPAIR Cypher queries using a SELF-REFINEMENT LOOP.

        Inputs: question, schema, previous_query, error_message

        If previous_query is empty → generate a query (Mode A).
        If previous_query is not empty → repair it using error_message (Mode B).
        The system will run EXPLAIN <query>; if it fails, you will be called again.

        ----------------------------------------
        MODE A — GENERATE
        ----------------------------------------
        1. Read the schema and produce a syntactically valid Cypher query that answers the question.
        2. Follow these rules:
           - Scholar names: use .knownName
           - Country/City/Continent/Institution: use .name
           - Use short variable names (s1, p1, c1, i1…)
           - Follow relationship directions exactly
           - String match: WHERE toLower(x) CONTAINS toLower("value")
           - No APOC
           - RETURN only properties or COUNT(...)
           - One Cypher statement, no newlines, comments, fences, EXPLAIN, or CALL
        3. Output ONLY the query.

        ----------------------------------------
        MODE B — REPAIR
        ----------------------------------------
        1. Use error_message to locate the issue:
           - Unknown label/property
           - Unbound variable
           - Wrong direction
           - Invalid function/type
           - Missing/invalid RETURN
        2. Compare previous_query against the schema and fix ONLY what is required.
        3. Preserve the meaning of the question.
           Prefer small local edits unless the query is fundamentally invalid.
        4. Ensure:
           - All labels and properties exist in schema
           - All variables are introduced in MATCH/WITH
           - Relationship directions and syntax are valid
        5. Follow all Mode A syntax rules.
        6. Output ONLY the repaired query.

        ----------------------------------------
        GLOBAL RULES
        ----------------------------------------
        - No explanations, comments, or natural language.
        - No code fences.
        - Output exactly ONE valid Cypher statement.
        - **Do NOT use ORDER BY, LIMIT, or SKIP under any circumstances.**
        """

        question: str = dspy.InputField()
        input_schema: str = dspy.InputField()
        previous_query: str | None = dspy.InputField()
        error_message: str | None = dspy.InputField()
        query: str = dspy.OutputField()


    class AnswerQuestion(dspy.Signature):
        """
        - Use the provided question, the generated Cypher query and the context to answer the question.
        - If the context is empty, state that you don't have enough information to answer the question.
        - When dealing with dates, mention the month in full.
        """

        question: str = dspy.InputField()
        cypher_query: str = dspy.InputField()
        context: str = dspy.InputField()
        response: str = dspy.OutputField()
    return

@app.cell
def _(dspy):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    lm = dspy.LM(
        "gemini/gemini-2.5-flash-lite",   
        api_key=GEMINI_API_KEY,
        temperature=0.0,
        max_tokens=1024,           
    )

    dspy.configure(lm=lm)
    return


@app.cell
def _(kuzu):
    class KuzuDatabaseManager:
        """Manages Kuzu database connection and schema retrieval."""

        def __init__(self, db_path: str = "ldbc_1.kuzu"):
            self.db_path = db_path
            self.db = kuzu.Database(db_path, read_only=True)
            self.conn = kuzu.Connection(self.db)

        @property
        def get_schema_dict(self) -> dict[str, list[dict]]:
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'NODE' RETURN *;")
            nodes = [row[1] for row in response]  # type: ignore
            response = self.conn.execute("CALL SHOW_TABLES() WHERE type = 'REL' RETURN *;")
            rel_tables = [row[1] for row in response]  # type: ignore
            relationships = []
            for tbl_name in rel_tables:
                response = self.conn.execute(f"CALL SHOW_CONNECTION('{tbl_name}') RETURN *;")
                for row in response:
                    relationships.append({"name": tbl_name, "from": row[0], "to": row[1]})  # type: ignore
            schema = {"nodes": [], "edges": []}

            for node in nodes:
                node_schema = {"label": node, "properties": []}
                node_properties = self.conn.execute(f"CALL TABLE_INFO('{node}') RETURN *;")
                for row in node_properties:  # type: ignore
                    node_schema["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["nodes"].append(node_schema)

            for rel in relationships:
                edge = {
                    "label": rel["name"],
                    "from": rel["from"],
                    "to": rel["to"],
                    "properties": [],
                }
                rel_properties = self.conn.execute(f"""CALL TABLE_INFO('{rel["name"]}') RETURN *;""")
                for row in rel_properties:  # type: ignore
                    edge["properties"].append({"name": row[1], "type": row[2]})  # type: ignore
                schema["edges"].append(edge)
            return schema
    return (KuzuDatabaseManager,)


@app.cell
def _(BaseModel, Field):
    class Query(BaseModel):
        query: str = Field(description="Valid Cypher query with no newlines")


    class Property(BaseModel):
        name: str
        type: str = Field(description="Data type of the property")


    class Node(BaseModel):
        label: str
        properties: list[Property] | None


    class Edge(BaseModel):
        label: str = Field(description="Relationship label")
        from_: Node = Field(alias="from", description="Source node label")
        to: Node = Field(alias="to", description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[Edge]
    return


@app.cell
def _():
    """Load few-shot examples for Text2Cypher from JSON file."""

    with open("data/few_shot_examples.json", "r", encoding="utf-8") as f:
        FEW_SHOT_EXAMPLES = json.load(f)

    return (FEW_SHOT_EXAMPLES,)


@app.cell
def _(FEW_SHOT_EXAMPLES,np):
    from sentence_transformers import SentenceTransformer
    
    class FewShotRetriever:
        def __init__(self, examples: list[dict], model_name: str = "all-MiniLM-L6-v2", k: int = 3):
            self.examples = examples
            self.k = k
            self.model = SentenceTransformer(model_name)
            
            # Pre-compute embeddings for all example questions
            example_questions = [ex["question"] for ex in examples]
            self.example_embeddings = self.model.encode(
                example_questions, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
        
        def retrieve(self, question: str) -> list[dict]:
            question_embedding = self.model.encode(
                [question], 
                convert_to_tensor=False,
                show_progress_bar=False
            )[0]
            
            similarities = []
            for ex_embedding in self.example_embeddings:
                similarity = np.dot(question_embedding, ex_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(ex_embedding)
                )
                similarities.append(similarity)
            
            top_k_indices = np.argsort(similarities)[-self.k:][::-1]
            
            return [self.examples[i] for i in top_k_indices]
        
        def format_examples_for_prompt(self, examples: list[dict]) -> str:
            """
            Format retrieved examples as a string for the prompt.

            """
            formatted = "Here are some similar examples:\n\n"
            for i, ex in enumerate(examples, 1):
                formatted += f"Example {i}:\n"
                formatted += f"Question: {ex['question']}\n"
                formatted += f"Cypher Query: {ex['cypher_query']}\n\n"
            return formatted
    
    return (FewShotRetriever,)


@app.cell
def _(
    AnswerQuestion,
    Any,
    FewShotRetriever,
    FEW_SHOT_EXAMPLES,
    KuzuDatabaseManager,
    PruneSchema,
    Text2Cypher,
    dspy,
):
    from profiler import Profiler
    from lru_cache import Text2CypherCache
    from performance_data import PerformanceDataExporter

    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.
        """

        def __init__(self, use_few_shot: bool = True, k_examples: int = 3, cache_size: int = 3):

            self.prune = dspy.Predict(PruneSchema)
            self.text2cypher = dspy.ChainOfThought(Text2Cypher)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)

            self.profiler = Profiler()

            self.cache = Text2CypherCache(max_size=cache_size)
            print(f"Text2Cypher cache initialized (max size: {cache_size})")

            self.use_few_shot = use_few_shot
            if use_few_shot:
                self.few_shot_retriever = FewShotRetriever(
                    examples=FEW_SHOT_EXAMPLES,
                    k=k_examples
                )
                print(f"Few-shot retriever initialized with {len(FEW_SHOT_EXAMPLES)} examples (retrieving top-{k_examples})")
            else:
                self.few_shot_retriever = None
                print("Running without few-shot examples")
            
            self.exporter = PerformanceDataExporter()
            self.query_counter = 0
            self.session_id = None

        def _get_session_filename(self):
            
            if self.session_id is None:
                from datetime import datetime
                self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"session_{self.session_id}"

        def _export_now(self):
            try:
                session_filename = self._get_session_filename()
                self.exporter.export_simple(session_filename)
            except Exception as e:
                print(f"Export error: {e}")
                import traceback
                traceback.print_exc()

        def __del__(self):
            try:
                if hasattr(self, 'query_counter') and self.query_counter > 0:
                    self._export_now()
            except Exception as e:
                import sys
                print(f"Warning during cleanup: {e}", file=sys.stderr)

        @staticmethod
        def _clean_cypher(query: str) -> str:
            """Remove ``` and ```cypher fences if the LM returns a code block."""
            q = query.strip()

            # Strip leading ```... fence
            if q.startswith("```"):
                # remove first ```
                q = q[3:]
                q = q.lstrip()

                if q.lower().startswith("cypher"):
                    q = q[len("cypher"):].lstrip()

            # Strip trailing ``` if present
            if q.endswith("```"):
                q = q[:-3]

            # If there is any remaining ``` (e.g. middle), cut at first
            if "```" in q:
                q = q.split("```", 1)[0]

            return q.strip()

        @staticmethod
        def postprocess_cypher(query: str) -> str:
            """
            Rule-based post-processor for Cypher queries.
            """
            q = query.strip()
            if not q:
                return q

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

            property_preferences = {
                "Scholar": ["knownName", "fullName"],
                "Prize": ["category", "awardYear", "prize_id"],
                "Institution": ["name"],
                "City": ["name", "state"],
                "Country": ["name"],
                "Continent": ["name"],
            }
            fallback_priority = [
                "knownName",
                "name",
                "category",
                "fullName",
                "gender",
                "motivation",
                "portion",
                "scholar_type",
            ]

            property_pattern = "|".join(re.escape(p) for p in sorted(text_properties))
            comparison_pattern = re.compile(
                "(?is)"
                "(toLower\\s*\\(\\s*)?"
                "([A-Za-z_]\\w*\\.(" + property_pattern + "))"
                "\\s*(\\))?"
                "\\s*(=|<>|!=|CONTAINS|STARTS\\s+WITH|ENDS\\s+WITH)"
                "\\s*('(?:[^'\\\\]|\\\\.)*'|\"(?:[^\\\"\\\\]|\\\\.)*\")"
            )

            def _normalize_comparison(match: re.Match[str]) -> str:
                property_expr = match.group(2)
                operator = " ".join(match.group(5).upper().split())
                literal = match.group(6).strip()
                normalized_prop = f"toLower({property_expr})"
                normalized_literal = f"toLower({literal})"
                return f"{normalized_prop} {operator} {normalized_literal}"

            q = comparison_pattern.sub(_normalize_comparison, q)

            var_labels: dict[str, str] = {}
            for m in re.finditer(r"\(\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w]*)", q):
                var, label = m.groups()
                var_labels.setdefault(var, label)

            var_properties: dict[str, list[str]] = {}
            for m in re.finditer(r"\b([A-Za-z_]\w*)\.(\w+)\b", q):
                var, prop = m.groups()
                props = var_properties.setdefault(var, [])
                if prop not in props:
                    props.append(prop)

            return_match = re.search(r"(?i)\breturn\b", q)
            if not return_match:
                return q

            prefix = q[: return_match.start()]
            body_and_tail = q[return_match.end() :]

            tail_start = len(body_and_tail)
            for pattern in (r"\bORDER\s+BY\b", r"\bLIMIT\b", r"\bSKIP\b"):
                match = re.search(pattern, body_and_tail, flags=re.IGNORECASE)
                if match and match.start() < tail_start:
                    tail_start = match.start()

            return_body = body_and_tail[:tail_start].strip()
            tail = body_and_tail[tail_start:]

            distinct = False
            if return_body.upper().startswith("DISTINCT "):
                distinct = True
                return_body = return_body[len("DISTINCT ") :].strip()

            def _split_return_items(clause: str) -> list[str]:
                items: list[str] = []
                current: list[str] = []
                depth = 0
                in_single = False
                in_double = False

                for ch in clause:
                    if ch == "'" and not in_double:
                        in_single = not in_single
                    elif ch == '"' and not in_single:
                        in_double = not in_double

                    if not in_single and not in_double:
                        if ch == "(":
                            depth += 1
                        elif ch == ")" and depth > 0:
                            depth -= 1

                    if (
                        ch == ","
                        and depth == 0
                        and not in_single
                        and not in_double
                    ):
                        items.append("".join(current).strip())
                        current = []
                    else:
                        current.append(ch)

                remainder = "".join(current).strip()
                if remainder:
                    items.append(remainder)

                return [item for item in items if item]

            def _select_projection(var: str) -> str | None:
                label = var_labels.get(var)
                used_props = var_properties.get(var, [])

                if label and label in property_preferences:
                    for candidate in property_preferences[label]:
                        if candidate in used_props:
                            return candidate
                    prefs = property_preferences[label]
                    if prefs:
                        return prefs[0]

                if used_props:
                    for preferred in fallback_priority:
                        if preferred in used_props:
                            return preferred
                    return used_props[0]

                return None

            def _rewrite_item(item: str) -> str:
                trimmed = item.strip()
                if not trimmed:
                    return trimmed

                alias_match = re.match(
                    r"(?is)(.+?)\s+AS\s+([A-Za-z_]\w*)$", trimmed
                )
                if alias_match:
                    expr = alias_match.group(1).strip()
                    alias = alias_match.group(2)
                else:
                    expr = trimmed
                    alias = None

                if re.fullmatch(r"[A-Za-z_]\w*", expr):
                    projection = _select_projection(expr)
                    if projection:
                        alias_name = alias or expr
                        return f"{expr}.{projection} AS {alias_name}"

                return trimmed

            items = _split_return_items(return_body)
            rewritten_items = [_rewrite_item(item) for item in items]

            new_return = ", ".join(rewritten_items)
            if distinct:
                new_return = f"DISTINCT {new_return}"

            return f"{prefix}RETURN {new_return}{tail}"

        def get_cypher_query(
            self,
            question: str,
            input_schema: str,
            previous_query: str | None = None,
            error_message: str | None = None,
        ) -> str:
            
            if previous_query is None and error_message is None:
                with self.profiler.profile("cache_lookup"):
                    cached_query = self.cache.get(question, input_schema)
                    if cached_query:
                        print(f"Cache hit for question: {question[:50]}...")
                        return cached_query

            with self.profiler.profile("schema_pruning"):
                prune_result = self.prune(question=question, input_schema=input_schema)
                schema = prune_result.pruned_schema

            previous_query_text = previous_query or ""
            error_text = error_message or ""

            if self.use_few_shot and self.few_shot_retriever:
                with self.profiler.profile("few_shot_retrieval"):
                    similar_examples = self.few_shot_retriever.retrieve(question)
                    examples_text = self.few_shot_retriever.format_examples_for_prompt(similar_examples)
                
                enhanced_schema = f"{examples_text}\n\nSchema:\n{str(schema)}"
        
                with self.profiler.profile("text2cypher_generation"):
                    text2cypher_result = self.text2cypher(
                        question=question,
                        input_schema=enhanced_schema,
                        previous_query=previous_query_text,
                        error_message=error_text,
                )
            else:
                # Generate without examples (original behavior)
                with self.profiler.profile("text2cypher_generation"):
                    text2cypher_result = self.text2cypher(
                        question=question,
                        input_schema=str(schema),
                        previous_query=previous_query_text,
                        error_message=error_text,
                )
            
            cypher_query: str = text2cypher_result.query
            # clean markdown fences
            with self.profiler.profile("post_processing"):
                cypher_query = self._clean_cypher(cypher_query)
                cypher_query = self.postprocess_cypher(cypher_query)

            # cache the query if this is a successful initial generation (not a repair)
            if previous_query is None and error_message is None:
                with self.profiler.profile("cache_store"):
                    self.cache.put(question, input_schema, cypher_query)

            return cypher_query

        def run_query(
            self, db_manager, question: str, input_schema: str
        ) -> tuple[str, list | None]:
            """
            Run a query synchronously on the database with a self-refinement loop.
            """
            max_attempts = 3
            previous_query = None
            error_message = None
            final_query = ""

            attempt_no = 0

            for _ in range(max_attempts):
                attempt_no += 1
                stage_name = f"text2cypher_attempt_{attempt_no}"
                with self.profiler.profile(stage_name):
                    candidate_query = self.get_cypher_query(
                        question=question,
                        input_schema=input_schema,
                        previous_query=previous_query,
                        error_message=error_message,
                )

                if not candidate_query:
                    break

                try:
                    with self.profiler.profile("query_validation"):
                        db_manager.conn.execute(f"EXPLAIN {candidate_query}")
                except Exception as e:
                    previous_query = candidate_query
                    error_message = str(e)
                    print("\n=== SELF-REFINEMENT TRIGGERED ===")
                    print("Previous query:", previous_query)
                    print("Error message:", error_message)
                    print("Entering repair mode...\n")
                    continue

                final_query = candidate_query
                break

            if not final_query:
                return previous_query or "", None

            try:
                with self.profiler.profile("query_execution"):
                    result = db_manager.conn.execute(final_query)
                    results = [item for row in result for item in row]
            except RuntimeError as e:
                print(f"Error running query: {e}")
                results = None

            

            return final_query, results

        def forward(self, db_manager, question: str, input_schema: str):

            self.query_counter += 1
            start_time = time.time()

            initial_cache_hits = self.cache.stats.hits

            with self.profiler.profile("total_pipeline"):
                with self.profiler.profile("query_phase"):
                    final_query, final_context = self.run_query(
                        db_manager, question, input_schema
                    )

                if final_context is None:
                    print(
                        "Empty results obtained from the graph database. Please retry with a different question."
                    )

                    total_time = time.time() - start_time
                    stage_times = {
                        stage_name: stage_metrics.last_duration
                        for stage_name, stage_metrics in self.profiler.stages.items()
                        if stage_metrics.durations
                    }

                    final_cache_hits = self.cache.stats.hits
                    cache_hit = (final_cache_hits > initial_cache_hits)

                    self.exporter.record_query(
                        query_no= self.query_counter,
                        question=question,
                        cache_hit=cache_hit,
                        total_time= total_time,
                        stage_time=stage_times
                    )

                    self._export_now()
                    return {
                        "question": question,
                        "query": final_query,
                        "answer": dspy.Prediction(response="No results from the graph."),
                }
                else:
                    with self.profiler.profile("answer_generation"):
                        answer = self.generate_answer(
                            question=question,
                            cypher_query=final_query,
                            context=str(final_context),
                )
                    
                    response = {
                        "question": question,
                        "query": final_query,
                        "answer": answer,
                    }

                    total_time = time.time() - start_time
                    stage_times = {
                        stage_name: stage_metrics.last_duration
                        for stage_name, stage_metrics in self.profiler.stages.items()
                        if stage_metrics.durations
                    }
                    final_cache_hits = self.cache.stats.hits
                    cache_hit = (final_cache_hits > initial_cache_hits)
                    self.exporter.record_query(
                        query_no= self.query_counter,
                        question=question,
                        cache_hit=cache_hit,
                        total_time= total_time,
                        stage_time=stage_times
                    )

                    print("\n")
                    self.print_timing_summary()
                    self._export_now()

                    return response

        async def aforward(self, db_manager, question: str, input_schema: str):
            final_query, final_context = self.run_query(
                db_manager, question, input_schema
            )
            if final_context is None:
                print(
                    "Empty results obtained from the graph database. Please retry with a different question."
                )
                return {
                    "question": question,
                    "query": final_query,
                    "answer": dspy.Prediction(response="No results from the graph."),
                }
            else:
                answer = self.generate_answer(
                    question=question,
                    cypher_query=final_query,
                    context=str(final_context),
                )
                response = {
                    "question": question,
                    "query": final_query,
                    "answer": answer,
                }
                return response
            
        def get_timing_results(self) -> dict:
            return self.profiler.get_summary_dict()
        
        def print_timing_summary(self) -> None:
            self.profiler.print_summary()
            self.cache.print_stats()

        def get_cache_stats(self) -> dict:
            stats = self.cache.get_stats()
            return{
                "cache_size": len(self.cache),
                "max_size": self.cache.max_size,
                "total_requests": stats.total_requests,
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "evictions": stats.evictions

            }


        def reset_timing(self) -> None:
            self.profiler.reset()
            self.cache.stats.reset()
        
        def export_performance_data(self, base_filename: str = None):

            profiler_data = self.profiler.get_summary_dict()
            cache_stats = {
                "total_requests": self.cache.stats.total_requests,
                "hits": self.cache.stats.hits,
                "misses": self.cache.stats.misses,
                "hit_rate": self.cache.stats.hit_rate,
                "evictions": self.cache.stats.evictions,
                "cache_size": len(self.cache.cache)
            }

            return self.exporter.export_all(profiler_data, cache_stats, base_filename)

    def run_graph_rag(questions: list[str], db_manager, use_few_shot: bool = True, k_examples: int = 3, cache_size: int = 128) -> list:
        """
        Run GraphRAG on a list of questions.
        """
        schema = str(db_manager.get_schema_dict)
        rag = GraphRAG(use_few_shot=use_few_shot, k_examples=k_examples, cache_size=cache_size)
        results = []
        for question in questions:
            response = rag(
                db_manager=db_manager,
                question=question,
                input_schema=schema,
            )
            results.append(response)
        return results, rag

    return (GraphRAG, run_graph_rag)


@app.cell
def _():
    return


@app.cell
def _():
    import json
    import marimo as mo
    import os
    import re
    import time
    from textwrap import dedent
    from typing import Any

    import dspy
    import kuzu
    import numpy as np
    from dotenv import load_dotenv
    from dspy.adapters.baml_adapter import BAMLAdapter
    from pydantic import BaseModel, Field

    load_dotenv()

    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return (
        Any,
        BAMLAdapter,
        BaseModel,
        Field,
        dspy,
        json,
        kuzu,
        mo,
        np,
        os,
        re,
    )



if __name__ == "__main__":
    app.run()
