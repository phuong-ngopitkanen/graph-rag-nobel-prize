# GraphRAG Few-shot Test Summary

- Total examples: 54
- EXPLAIN success: 53 (first-pass: 50, refinements: 3)
- EXPLAIN failures after retries: 1
- Execution failures: 2
- Post-processor fully passing: 43

## Problematic Examples

### EXPLAIN failures
- Which laureates won a prize before the age of 40? — Catalog exception: function TOINTEGER does not exist.

### Post-processor issues (lowercase)

- None

### Post-processor issues (RETURN projection)

- Which scholars have won more than one Nobel Prize?
- For each prize category, how many distinct laureates are there?
- Which cities have produced at least five laureates by birth?
- For each continent, how many Physics laureates were born there?
- Which laureates have been affiliated with more than one institution?
- What is the average adjusted prize amount per category for prizes awarded from 1980 onwards?
- For each continent, how many distinct laureates have affiliated institutions there?
- For each continent, how many laureates have died there?
- Which scholars have affiliations in at least three different countries?
- For each birth continent, how many laureates later worked at institutions on a different continent?
- What is the average height of physics laureates?

### Execution failures

- How many laureates in Chemistry were born before 1900? — Conversion exception: Cast failed. Could not convert "1926-08-11" to INT64.
- Which laureates won a prize before the age of 40? — Catalog exception: function TOINTEGER does not exist.

## Strengths

- Single-attempt successes: 50
- Successful repairs after EXPLAIN feedback: 3