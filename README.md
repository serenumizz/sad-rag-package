# Ask My SAD Project — RAG + Tool Use + Eval

Grounded Q&A over my final project — *Temperature and
Light Effects on Mood Trends* (Reddit sentiment in r/nyc vs. NYC weather,
Jan 1 – May 9 2025, ~87.7k scored posts), built as a lens on Seasonal
Affective Disorder (SAD).

Three pieces, each mapping directly to a JD requirement:

| Piece | File | What it proves |
|---|---|---|
| RAG pipeline | `src/ingest.py`, `src/retrieve.py` | embed docs, retrieve relevant chunks, answer grounded in them |
| Tool use | `src/tools.py`, `src/build_db.py` | model calls a real function — SQL against the actual merged dataset — instead of answering from memory |
| Eval harness | `src/eval.py` | automated test questions with expected answers, scored pass/fail |

## How it's grounded

- **RAG knowledge base** (`data/project_docs.py`): the project's own
  methodology, 5 analysis Q&As, and summary of findings, pulled straight out
  of the notebook's markdown cells.
- **Tool-use data** (`data/sad.db`): built from the *actual* project data —
  87,725 individually-scored Reddit posts/comments
  (`analyzed_sentiment_gpt_like.csv`) merged with daily NYC weather
  (`weather_openmeteo_...csv`), exactly the way the notebook's own
  `groupby(["date","subreddit"])["score"].mean()` + merge does it.
- One real bug was caught and fixed along the way: the weather CSV's saved
  `light_intensity` column was stale (it labeled everything "low" even past
  a radiation sum of 29, which the notebook's own bins would call "high").
  `build_db.py` recomputes it from `shortwave_radiation_sum` using the
  notebook's own bins instead of trusting the saved column.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
export $(cat .env | xargs)
```

## Run it

```bash
# 1. Build the SQL database from the real data (one-time)
python src/build_db.py \
  --sentiment /path/to/analyzed_sentiment_gpt_like.csv \
  --weather /path/to/weather_openmeteo_*.csv

# 2. Embed the knowledge-base docs (one-time, or whenever project_docs.py changes)
python src/ingest.py

# 3. Ask questions
python src/ask.py "What did the report conclude about temperature and mood?"
python src/ask.py --verbose "What is the average sentiment on high light-intensity days?"

# 4. Run the eval suite
python src/eval.py
```

## How a question gets answered

1. `ask.py` embeds the question and retrieves the top-3 most similar chunks
   from the knowledge base (`retrieve.py`).
2. Both the retrieved context and the question go to Claude, along with the
   `query_sentiment_weather` tool definition.
3. Claude decides: if the answer is already in the retrieved write-up
   (e.g. "what did the report recommend for further research?"), it answers
   from that. If the question needs a real number not already stated
   verbatim (e.g. "what's the average sentiment on medium-light days?"), it
   calls the tool, which runs real SQL against `sad.db` and returns rows.
4. The final answer says which source it used.

## Eval design

10 questions, 4 requiring the RAG path and 6 requiring the tool-use path
(`src/eval.py`). Numeric answers are graded by checking whether a number
within tolerance of the expected value appears in the response; text
answers are graded by required-keyword presence. Running it prints a
pass/fail per question plus an overall score, split by source (RAG vs.
tool), e.g.:

```
Overall:    9/10 passed (90%)
RAG-only:   4/4 passed
Tool-use:   5/6 passed
```

## Resume bullet

> Built a retrieval-augmented generation (RAG) pipeline grounded in a prior
> data-science project (87k-row sentiment/weather dataset), with a
> function-calling tool step for live SQL queries against the real data and
> an automated evaluation harness scoring both the retrieval and tool-use
> paths.
