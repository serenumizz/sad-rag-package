"""
Builds the SQLite database the tool-use step queries against.

Merges the real per-post sentiment data (87,725 scored Reddit posts/comments,
r/nyc, Jan 1 - May 9 2025) with daily NYC weather (temperature + light
intensity), matching the notebook's own merge logic:
    avg_daily_sentiment = sentiment_df.groupby(["date","subreddit"])["score"].mean()
    merge_df = avg_daily_sentiment.merge(weather_df, on="date"/"city")

Produces two tables in data/sad.db:
  - daily_sentiment_weather: one row per day (date, subreddit, avg_score,
      n_posts, avg_temp, shortwave_radiation_sum, light_intensity)
  - posts: the raw per-post rows (id, date, sentiment, emotion, score, text)
      so the tool can also answer finer-grained questions if needed.

Run:
    python src/build_db.py \
        --sentiment /path/to/analyzed_sentiment_gpt_like.csv \
        --weather /path/to/weather_openmeteo_*.csv
"""
import argparse
import os
import sqlite3

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sad.db")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", required=True, help="path to analyzed_sentiment_gpt_like.csv")
    parser.add_argument("--weather", required=True, help="path to weather_openmeteo_*.csv")
    parser.add_argument("--out", default=DB_PATH)
    args = parser.parse_args()

    sent = pd.read_csv(args.sentiment)
    weather = pd.read_csv(args.weather)

    # The project's own CSV has a stale `light_intensity` column (it marks
    # everything "low" even past a radiation sum of 29, which the notebook's
    # own bins would call "high"). Recompute it from shortwave_radiation_sum
    # using the exact bins/labels the notebook defines, so downstream
    # queries aren't grounded in a bug.
    weather["light_intensity"] = pd.cut(
        weather["shortwave_radiation_sum"],
        bins=[-float("inf"), 10, 20, float("inf")],
        labels=["low", "medium", "high"],
    )

    # Daily aggregate per subreddit, matching the notebook's own groupby
    daily = (
        sent.groupby(["date", "subreddit"], as_index=False)
        .agg(avg_score=("score", "mean"), n_posts=("score", "count"))
    )

    merged = daily.merge(
        weather, left_on=["date", "subreddit"], right_on=["date", "city"]
    ).drop(columns=["city"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    conn = sqlite3.connect(args.out)

    merged.to_sql("daily_sentiment_weather", conn, if_exists="replace", index=False)
    sent[["id", "subreddit", "date", "sentiment", "emotion", "score"]].to_sql(
        "posts", conn, if_exists="replace", index=False
    )

    conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_sentiment_weather(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(date)")
    conn.commit()

    n_daily = conn.execute("SELECT COUNT(*) FROM daily_sentiment_weather").fetchone()[0]
    n_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    conn.close()

    print(f"Wrote {args.out}")
    print(f"  daily_sentiment_weather: {n_daily} rows")
    print(f"  posts: {n_posts} rows")


if __name__ == "__main__":
    main()
