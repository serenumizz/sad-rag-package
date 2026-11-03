"""
Step 2 of the pipeline: tool use.

Instead of answering numeric/statistical questions from retrieved text (which
risks the LLM paraphrasing or misremembering a number), the model can call
`query_sentiment_weather`, a real function that runs SQL against the actual
merged dataset (data/sad.db, built by build_db.py from the project's own
87,725-row sentiment file and the NYC weather file).

This is the "tool use" / function-calling piece: the model decides whether a
question needs this (e.g. "what was avg sentiment in March?") vs. the RAG
text (e.g. "what did the report conclude?").
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sad.db")

# JSON schema describing the tool to the LLM (OpenAI / Anthropic tool-use format)
TOOL_SCHEMA = {
    "name": "query_sentiment_weather",
    "description": (
        "Run a read-only SQL query against the project's real merged dataset. "
        "Table `daily_sentiment_weather` has one row per day: "
        "date (YYYY-MM-DD), subreddit, avg_score (mean Reddit sentiment score "
        "that day), n_posts, avg_temp (Celsius), shortwave_radiation_sum, "
        "light_intensity (low/medium/high). Table `posts` has one row per "
        "Reddit post/comment: id, subreddit, date, sentiment (positive/"
        "negative/neutral label), emotion, score. Use this for any question "
        "needing a real number, aggregate, filter, or comparison over the "
        "actual data, rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single read-only SELECT statement (SQLite syntax).",
            }
        },
        "required": ["sql"],
    },
}


def query_sentiment_weather(sql: str) -> dict:
    """Execute a read-only SELECT against sad.db and return rows as dicts."""
    if not sql.strip().lower().startswith("select"):
        return {"error": "Only SELECT statements are allowed."}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        return {"rows": rows, "row_count": len(rows)}
    except sqlite3.Error as e:
        return {"error": str(e)}
    finally:
        conn.close()


if __name__ == "__main__":
    # quick manual smoke test
    print(query_sentiment_weather(
        "SELECT light_intensity, AVG(avg_score) as mean_sentiment, COUNT(*) as n_days "
        "FROM daily_sentiment_weather GROUP BY light_intensity"
    ))
