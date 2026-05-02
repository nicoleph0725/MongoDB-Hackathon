import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
from langchain_fireworks import ChatFireworks
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DB_NAME")]

# Collections
trends_collection   = db["google_trends"]
memes_collection    = db["meme_posts"]
analysis_collection = db["analysis_results"]

# Configure Fireworks via LangChain
llm = ChatFireworks(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    model="accounts/fireworks/models/deepseek-v4-pro",
    temperature=0.7,
    max_tokens=2048
)


# ─────────────────────────────────────────
# PART 1 — Analyse Google Trends
# ─────────────────────────────────────────

trends_prompt = PromptTemplate(
    input_variables=["trends_data"],
    template="""
    You are an expert internet culture analyst specialising in meme trends.

    Below is Google Trends data showing interest levels for meme-related keywords over the last 24 hours.
    Interest is scored out of 100 where 100 means peak popularity.

    {trends_data}

    Analyse this data and provide:
    1. Which keywords are currently peaking and why they might be trending
    2. Which keywords are declining and may be dying out
    3. Any patterns you notice across the keywords
    4. What this tells us about the current state of internet culture

    Be specific, insightful, and use internet culture knowledge in your analysis.
    """
)

def analyse_trends():
    print("🔍 Analysing Google Trends data...\n")

    trends = list(trends_collection.find({}, {"_id": 0}))

    if not trends:
        print("⚠️ No trends data found. Run scraper.py first!")
        return

    trends_data = "\n".join([
        f"- Keyword: '{t['keyword']}' | Avg Interest: {t['avg_interest']}/100 | Peak: {t['max_interest']}/100"
        for t in trends
    ])

    chain    = trends_prompt | llm
    response = chain.invoke({"trends_data": trends_data})
    analysis = response.content

    analysis_collection.insert_one({
        "type":        "trends_analysis",
        "analysis":    analysis,
        "analysed_at": datetime.now(tz=timezone.utc),
        "source":      "google_trends"
    })

    print("📊 Trends Analysis:")
    print("─" * 50)
    print(analysis)
    print("─" * 50)
    print("✅ Trends analysis saved to MongoDB!\n")


# ─────────────────────────────────────────
# PART 2 — Analyse Know Your Meme Posts
# ─────────────────────────────────────────

memes_prompt = PromptTemplate(
    input_variables=["memes_data"],
    template="""
    You are an expert internet culture analyst who lives and breathes meme culture.

    Below are the latest and most popular memes from Know Your Meme right now:

    {memes_data}

    Analyse this data and provide:
    1. What meme formats are currently dominating
    2. What themes and topics are recurring across multiple memes
    3. Which memes seem fresh and emerging vs which seem oversaturated
    4. What does Gen Z and younger internet culture seem obsessed with right now

    Be specific and use your knowledge of internet culture to give deep insights.
    """
)

def analyse_memes():
    print("🔍 Analysing Know Your Meme data...\n")

    memes = list(memes_collection.find({}, {"_id": 0}).sort("scraped_at", -1).limit(30))

    if not memes:
        print("⚠️ No meme data found. Run scraper.py first!")
        return

    memes_data = "\n".join([
        f"- [{m['source']}] {m['title']} | {m.get('summary', '')[:100]}"
        for m in memes
    ])

    chain    = memes_prompt | llm
    response = chain.invoke({"memes_data": memes_data})
    analysis = response.content

    analysis_collection.insert_one({
        "type":        "memes_analysis",
        "analysis":    analysis,
        "analysed_at": datetime.now(tz=timezone.utc),
        "source":      "know_your_meme"
    })

    print("🎭 Meme Format Analysis:")
    print("─" * 50)
    print(analysis)
    print("─" * 50)
    print("✅ Meme analysis saved to MongoDB!\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("🧠 Starting Meme Analyst Agent...\n")
    analyse_trends()
    analyse_memes()
    print("🎉 Analysis complete! Check MongoDB for results.")