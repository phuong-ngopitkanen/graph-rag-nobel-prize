import marimo
import os

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
def _(KuzuDatabaseManager, mo, run_graph_rag, text_ui):
    db_name = "nobel.kuzu"
    db_manager = KuzuDatabaseManager(db_name)

    question = text_ui.value

    with mo.status.spinner(title="Generating answer...") as _spinner:
        result = run_graph_rag([question], db_manager)[0]

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
        "gemini/gemini-2.0-flash",   
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
        to: Node = Field(alias="from", description="Target node label")
        properties: list[Property] | None


    class GraphSchema(BaseModel):
        nodes: list[Node]
        edges: list[Edge]
    return


@app.cell
def _():
    
    """
    Few-shot examples for Text2Cypher with different query patterns
    
    """

    FEW_SHOT_EXAMPLES = [
        {
            "question": "Which scholars won prizes in Physics and were affiliated with University of Cambridge?",
            "cypher_query": "MATCH (s:Scholar)-[r1:WON]->(p:Prize) WHERE toLower(p.category) CONTAINS toLower('Physics') MATCH (s)-[r2:AFFILIATED_WITH]->(i:Institution) WHERE toLower(i.name) CONTAINS toLower('University of Cambridge') RETURN s.fullName"
        },
        {
            "question": "How many scholars won prizes in Chemistry?",
            "cypher_query": "MATCH (s:Scholar)-[r:WON]->(p:Prize) WHERE toLower(p.category) CONTAINS toLower('Chemistry') RETURN COUNT(DISTINCT s) AS count"
        },
        {
            "question": "Who were the mentors of Marie Curie?",
            "cypher_query": "MATCH (s:Scholar)-[r:MENTORED_BY]->(m:Scholar) WHERE toLower(s.knownName) CONTAINS toLower('Marie Curie') RETURN m.fullName"
        },
        {
            "question": "Which scholars were affiliated with MIT?",
            "cypher_query": "MATCH (s:Scholar)-[r:AFFILIATED_WITH]->(i:Institution) WHERE toLower(i.name) CONTAINS toLower('MIT') RETURN s.fullName"
        },
        {
            "question": "List all scholars who won prizes in Medicine",
            "cypher_query": "MATCH (s:Scholar)-[r:WON]->(p:Prize) WHERE toLower(p.category) CONTAINS toLower('Medicine') RETURN s.fullName"
        },
        {
            "question": "How many laureates were born in Germany?",
            "cypher_query": "MATCH (s:Scholar)-[r:BORN_IN]->(c:Country) WHERE toLower(c.name) CONTAINS toLower('Germany') RETURN COUNT(DISTINCT s) AS count"
        },
        {
            "question": "Which scholars won prizes and were born in the United States?",
            "cypher_query": "MATCH (s:Scholar)-[r1:WON]->(p:Prize) MATCH (s)-[r2:BORN_IN]->(c:Country) WHERE toLower(c.name) CONTAINS toLower('United States') RETURN s.fullName"
        },
        {
            "question": "Find all scholars affiliated with Harvard University",
            "cypher_query": "MATCH (s:Scholar)-[r:AFFILIATED_WITH]->(i:Institution) WHERE toLower(i.name) CONTAINS toLower('Harvard') RETURN s.fullName"
        },
        {
            "question": "Which scholars won prizes in both Physics and Chemistry?",
            "cypher_query": "MATCH (s:Scholar)-[r1:WON]->(p1:Prize) WHERE toLower(p1.category) CONTAINS toLower('Physics') MATCH (s)-[r2:WON]->(p2:Prize) WHERE toLower(p2.category) CONTAINS toLower('Chemistry') RETURN s.fullName"
        },
        {
            "question": "Count the number of scholars affiliated with University of Oxford",
            "cypher_query": "MATCH (s:Scholar)-[r:AFFILIATED_WITH]->(i:Institution) WHERE toLower(i.name) CONTAINS toLower('Oxford') RETURN COUNT(DISTINCT s) AS count"
        },
        {
            "question": "Who were the students of Niels Bohr?",
            "cypher_query": "MATCH (s:Scholar)-[r:MENTORED_BY]->(m:Scholar) WHERE toLower(m.knownName) CONTAINS toLower('Niels Bohr') RETURN s.fullName"
        },
        {
            "question": "Which scholars won prizes in Literature?",
            "cypher_query": "MATCH (s:Scholar)-[r:WON]->(p:Prize) WHERE toLower(p.category) CONTAINS toLower('Literature') RETURN s.fullName"
        },
        {
            "question": "Find scholars affiliated with institutions in Cambridge",
            "cypher_query": "MATCH (s:Scholar)-[r1:AFFILIATED_WITH]->(i:Institution)-[r2:LOCATED_IN]->(ci:City) WHERE toLower(ci.name) CONTAINS toLower('Cambridge') RETURN s.fullName"
        },
        {
            "question": "How many prizes were won by scholars born in France?",
            "cypher_query": "MATCH (s:Scholar)-[r1:BORN_IN]->(c:Country) WHERE toLower(c.name) CONTAINS toLower('France') MATCH (s)-[r2:WON]->(p:Prize) RETURN COUNT(p) AS count"
        },
        {
            "question": "Which scholars were mentored by Albert Einstein?",
            "cypher_query": "MATCH (s:Scholar)-[r:MENTORED_BY]->(m:Scholar) WHERE toLower(m.knownName) CONTAINS toLower('Albert Einstein') RETURN s.fullName"
        }
    ]
    
    return (FEW_SHOT_EXAMPLES,)


@app.cell
def _(FEW_SHOT_EXAMPLES,np):
    from sentence_transformers import SentenceTransformer
    
    class FewShotRetriever:
        """Retrieves most similar few-shot examples based on semantic similarity"""
        
        def __init__(self, examples: list[dict], model_name: str = "all-MiniLM-L6-v2", k: int = 3):
            """
            Initialize the retriever with examples and embedding model.
            
            Args:
                examples: List of dicts with 'question' and 'cypher_query' keys
                model_name: Name of sentence-transformers model to use
                k: Number of examples to retrieve
            """
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
            """
            Retrieve top-k most similar examples for the given question.
            
            Args:
                question: Input question to find similar examples for
                
            Returns:
                List of top-k most similar example dicts
            """
            # Encode the input question
            question_embedding = self.model.encode(
                [question], 
                convert_to_tensor=False,
                show_progress_bar=False
            )[0]
            
            # Compute cosine similarity with all examples
            similarities = []
            for ex_embedding in self.example_embeddings:
                similarity = np.dot(question_embedding, ex_embedding) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(ex_embedding)
                )
                similarities.append(similarity)
            
            # Get indices of top-k most similar examples
            top_k_indices = np.argsort(similarities)[-self.k:][::-1]
            
            # Return the top-k examples
            return [self.examples[i] for i in top_k_indices]
        
        def format_examples_for_prompt(self, examples: list[dict]) -> str:
            """
            Format retrieved examples as a string for the prompt.
            
            Args:
                examples: List of example dicts
                
            Returns:
                Formatted string of examples
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
    class GraphRAG(dspy.Module):
        """
        DSPy custom module that applies Text2Cypher to generate a query and run it
        on the Kuzu database, to generate a natural language response.
        """

        def __init__(self, use_few_shot: bool = True, k_examples: int = 3):
            """
            Initialize GraphRAG module.
            
            Args:
                use_few_shot: Whether to use dynamic few-shot examples
                k_examples: Number of examples to retrieve if use_few_shot is True
            """
            self.prune = dspy.Predict(PruneSchema)
            self.text2cypher = dspy.ChainOfThought(Text2Cypher)
            self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
            
            # Initialize few-shot retriever if enabled
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

        def get_cypher_query(
            self,
            question: str,
            input_schema: str,
            previous_query: str | None = None,
            error_message: str | None = None,
        ) -> str:
            """
            Generate Cypher query with optional dynamic few-shot examples.
            
            Args:
                question: Natural language question
                input_schema: Graph schema string
                previous_query: Previously generated query when repairing
                error_message: Error text from EXPLAIN when repairing
                
            Returns:
                Generated Cypher query string
            """
            # Prune schema
            prune_result = self.prune(question=question, input_schema=input_schema)
            schema = prune_result.pruned_schema
            previous_query_text = previous_query or ""
            error_text = error_message or ""

            # Retrieve and format few-shot examples if enabled
            if self.use_few_shot and self.few_shot_retriever:
                similar_examples = self.few_shot_retriever.retrieve(question)
                examples_text = self.few_shot_retriever.format_examples_for_prompt(similar_examples)
                
                # Add examples to the schema context
                enhanced_schema = f"{examples_text}\n\nSchema:\n{str(schema)}"
                
                # Generate Cypher query with examples
                text2cypher_result = self.text2cypher(
                    question=question,
                    input_schema=enhanced_schema,
                    previous_query=previous_query_text,
                    error_message=error_text,
                )
            else:
                # Generate without examples (original behavior)
                text2cypher_result = self.text2cypher(
                    question=question,
                    input_schema=str(schema),
                    previous_query=previous_query_text,
                    error_message=error_text,
                )
            
            cypher_query: str = text2cypher_result.query
            # clean markdown fences
            cypher_query = self._clean_cypher(cypher_query)
            return cypher_query

        def run_query(
            self, db_manager, question: str, input_schema: str
        ) -> tuple[str, list | None]:
            """
            Run a query synchronously on the database with a self-refinement loop.
            """
            max_attempts = 3
            previous_query = ""
            error_message = ""
            final_query = ""

            for _ in range(max_attempts):
                candidate_query = self.get_cypher_query(
                    question=question,
                    input_schema=input_schema,
                    previous_query=previous_query,
                    error_message=error_message,
                )

                if not candidate_query:
                    break

                try:
                    db_manager.conn.execute(f"EXPLAIN {candidate_query}")
                except Exception as e:
                    previous_query = candidate_query
                    error_message = str(e)
                    print("EXPLAIN error:", error_message)
                    continue

                final_query = candidate_query
                break

            if not final_query:
                return previous_query or "", None

            try:
                result = db_manager.conn.execute(final_query)
                results = [item for row in result for item in row]
            except RuntimeError as e:
                print(f"Error running query: {e}")
                results = None

            return final_query, results

        def forward(self, db_manager, question: str, input_schema: str):
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

    def run_graph_rag(questions: list[str], db_manager, use_few_shot: bool = True, k_examples: int = 3) -> list:
        """
        Run GraphRAG on a list of questions.
        
        Args:
            questions: List of questions to answer
            db_manager: Database manager instance
            use_few_shot: Whether to use dynamic few-shot examples
            k_examples: Number of examples to retrieve
            
        Returns:
            List of response dicts
        """
        schema = str(db_manager.get_schema_dict)
        rag = GraphRAG(use_few_shot=use_few_shot, k_examples=k_examples)
        results = []
        for question in questions:
            response = rag(
                db_manager=db_manager,
                question=question,
                input_schema=schema,
            )
            results.append(response)
        return results

    return (GraphRAG, run_graph_rag)


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import os
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
        kuzu,
        mo,
        np,
        os,
    )



if __name__ == "__main__":
    app.run()
