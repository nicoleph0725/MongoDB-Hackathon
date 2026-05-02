import os
import traceback
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
from langchain_fireworks import ChatFireworks
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client[os.getenv("DB_NAME")]

# Collections
analysis_collection    = db["analysis_results"]
predictions_collection = db["predictions"]

# Configure Fireworks via LangChain
llm = ChatFireworks(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    model="accounts/fireworks/models/deepseek-v4-pro",
    temperature=0.7,
    max_tokens=2048
)


# ─────────────────────────────────────────
# PART 1 — Load Previous Analyses
# ─────────────────────────────────────────

def load_analyses():
    print("📂 Loading analyses from MongoDB...\n")

    all_docs = list(analysis_collection.find({}))

    trends_docs = [d for d in all_docs if d.get("type") == "trends_analysis"]
    memes_docs  = [d for d in all_docs if d.get("type") == "memes_analysis"]

    if not trends_docs:
        print("⚠️ No trends analysis found! Run analyst.py first.")
        return None, None

    if not memes_docs:
        print("⚠️ No memes analysis found! Run analyst.py first.")
        return None, None

    trends_analysis = sorted(trends_docs, key=lambda x: x["analysed_at"], reverse=True)[0]
    memes_analysis  = sorted(memes_docs,  key=lambda x: x["analysed_at"], reverse=True)[0]

    trends_text = trends_analysis.get("analysis", "")
    memes_text  = memes_analysis.get("analysis", "")

    if not memes_text:
        for doc in sorted(memes_docs, key=lambda x: x["analysed_at"], reverse=True):
            if doc.get("analysis"):
                memes_text = doc["analysis"]
                break

    if not trends_text or not memes_text:
        print("⚠️ Analysis documents exist but content is empty. Re-run analyst.py.")
        return None, None

    print(f"✅ Loaded trends analysis ({len(trends_text)} chars)")
    print(f"✅ Loaded memes analysis ({len(memes_text)} chars)\n")

    return trends_text, memes_text


# ─────────────────────────────────────────
# PART 2 — Generate Predictions
# ─────────────────────────────────────────

predictor_prompt = PromptTemplate(
    input_variables=["trends_analysis", "memes_analysis"],
    template="""
    You are the world's best meme trend forecaster with encyclopedic knowledge
    of internet culture, meme history, and viral content patterns.

    Below are two analyses of current meme culture:

    GOOGLE TRENDS ANALYSIS:
    {trends_analysis}

    KNOW YOUR MEME ANALYSIS:
    {memes_analysis}

    Generate a detailed meme trend forecast report with:

    TOP 3 MEMES ABOUT TO GO VIRAL
    For each one:
    - Name/description of the meme format
    - Why it's about to blow up
    - What platforms it will dominate
    - How long it will last
    - Confidence level: High/Medium/Low

    TOP 3 MEMES THAT ARE DYING
    For each one:
    - Name of the meme/trend
    - Why it's losing steam
    - What killed it

    ONE WILDCARD PREDICTION
    - An unexpected trend that could come from nowhere
    - Why you think this could happen

    CONTENT CREATOR TIP OF THE DAY
    - One actionable tip for going viral this week

    Be bold, specific, and confident. Don't be generic.
    """
)

def generate_predictions(trends_analysis, memes_analysis):
    print("🔮 Generating meme trend predictions...\n")

    try:
        chain    = predictor_prompt | llm
        response = chain.invoke({
            "trends_analysis": trends_analysis,
            "memes_analysis":  memes_analysis
        })
        prediction = response.content

        predictions_collection.insert_one({
            "prediction":   prediction,
            "predicted_at": datetime.now(tz=timezone.utc),
            "based_on": {
                "trends_analysis": trends_analysis[:200],
                "memes_analysis":  memes_analysis[:200]
            }
        })

        return prediction

    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────
# PART 3 — Print The Report
# ─────────────────────────────────────────

def print_report(prediction):
    print("\n")
    print("=" * 60)
    print("       🌐 MEME TREND FORECAST REPORT")
    print(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(prediction)
    print("=" * 60)
    print("✅ Prediction saved to MongoDB!")
    print("=" * 60)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting Meme Predictor Agent...\n")

    trends_analysis, memes_analysis = load_analyses()

    if trends_analysis and memes_analysis:
        prediction = generate_predictions(trends_analysis, memes_analysis)
        if prediction:
            print_report(prediction)
        else:
            print("❌ Prediction generation failed. Check errors above.")
    else:
        print("❌ Cannot load analysis data. Re-run analyst.py first.")