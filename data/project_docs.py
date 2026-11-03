"""
Knowledge-base documents for the SAD (Seasonal Affective Disorder) RAG project.

These are derived from Yusri Ilza Sania's NYU "Dealing with Data" final project
notebook: "Temperature and Light Effects on Mood Trends" (Reddit sentiment vs.
NYC weather, Jan 1 - May 9, 2025).

Each entry is one retrievable document. Keeping them short and topic-focused
(rather than one giant blob) makes retrieval much more precise.
"""

DOCUMENTS = [
    {
        "id": "overview",
        "title": "Project Overview",
        "text": (
            "Project: Temperature and Light Effects on Mood Trends, by Yusri Ilza Sania. "
            "Objective: analyze the correlation between mood/sentiment and weather "
            "(temperature and light intensity), in the context of Seasonal Affective "
            "Disorder (SAD). Data covers New York City, January 1 to May 9, 2025, "
            "with approximately 87,000 observations."
        ),
    },
    {
        "id": "data_sources",
        "title": "Data Sources and Methodology",
        "text": (
            "Reddit posts and comments were pulled from the PullPush API "
            "(api.pullpush.io/reddit/search). Weather data (temperature and shortwave "
            "radiation, used as a proxy for light intensity) came from the Open-Meteo "
            "archive API (archive-api.open-meteo.com/v1/archive) for New York City "
            "(lat 40.7128, lon -74.0060). Sentiment analysis on Reddit text was performed "
            "using Google Cloud Platform's Natural Language Understanding (NLU) API. "
            "OpenAI's API (gpt-4.1-mini) was used to help generate the written report "
            "sections (introduction, summary, conclusion, title). Light intensity was "
            "bucketed into low/medium/high categories based on shortwave radiation sum "
            "(low: <=10 MJ/m^2, medium: 10-20, high: >20)."
        ),
    },
    {
        "id": "q1_temperature",
        "title": "Question 1: Temperature vs Sentiment",
        "text": (
            "Question: As temperature increases from winter to spring, does sentiment "
            "improve? Answer: Yes. Sentiment trends increase along with increasing "
            "temperature over the observed period."
        ),
    },
    {
        "id": "q2_light",
        "title": "Question 2: Light Intensity vs Sentiment",
        "text": (
            "Question: As light intensity increases, does sentiment improve? Answer: "
            "Yes. Sentiment trends increase along with increasing light intensity "
            "(shortwave radiation)."
        ),
    },
    {
        "id": "q3_avg_temp_positive",
        "title": "Question 3: Average Temperature on Positive-Sentiment Days",
        "text": (
            "Question: What is the average temperature when the emotional (sentiment) "
            "score is above 0? Answer: approximately 9.96 degrees Celsius "
            "(9.964444444444444)."
        ),
    },
    {
        "id": "q4_avg_temp_negative",
        "title": "Question 4: Average Temperature on Negative-Sentiment Days",
        "text": (
            "Question: What is the average temperature when the emotional (sentiment) "
            "score is below 0? Answer: approximately 2.99 degrees Celsius "
            "(2.985714285714286)."
        ),
    },
    {
        "id": "q5_overall_avg_sentiment",
        "title": "Question 5: Overall Average Sentiment Score",
        "text": (
            "Question: What is the average emotional score across the whole study "
            "period, January 1 to May 9, 2025? Answer: approximately -0.011 "
            "(-0.01102125596275067), which is slightly negative / just below neutral."
        ),
    },
    {
        "id": "key_findings",
        "title": "Summary of Key Findings",
        "text": (
            "The data covers January 1 to May 9, 2025 in New York, examining how "
            "temperature and light intensity relate to sentiment, as a lens on "
            "Seasonal Affective Disorder (SAD). Finding 1 - Warmer weather links to "
            "better mood: average temperature on positive-sentiment days was about "
            "9.96C, versus about 2.99C on negative-sentiment days, a roughly 7-degree "
            "gap. Finding 2 - Overall mood leaned slightly negative: the average "
            "sentiment score for the full period was -0.011, just below neutral, "
            "consistent with SAD patterns where the coldest months suppress mood "
            "before a gradual spring recovery. Finding 3 - Both light and temperature "
            "rise alongside sentiment: shortwave radiation and temperature trend "
            "upward from January through May, and sentiment becomes less volatile and "
            "more consistently positive by April-May, mirroring a seasonal recovery "
            "pattern tied to more daylight and warmth."
        ),
    },
    {
        "id": "further_research",
        "title": "Further Research Directions",
        "text": (
            "Three suggested directions for extending this project: (1) Lag effect "
            "analysis - test for a time lag between temperature/light changes and "
            "sentiment changes, to see whether mood improves immediately with better "
            "weather or only after a delay of days or weeks. (2) Extending the "
            "dataset - the current study ends in early May; tracking through summer "
            "and back into winter would allow a full seasonal cycle analysis and "
            "stronger conclusions about the SAD pattern. (3) Negative sentiment "
            "outliers - there are sharp sentiment dips even in warmer months (visible "
            "in May); understanding their causes could reveal that SAD is not purely "
            "driven by environmental factors."
        ),
    },
]
