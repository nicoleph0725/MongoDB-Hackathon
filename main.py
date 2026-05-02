import os
import time
import json
import traceback
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone
from langchain_fireworks import ChatFireworks
from langchain_core.prompts import PromptTemplate
from agent.scraper import scrape
from agent.analyst import analyse_trends, analyse_memes
from agent.predictor import load_analyses, generate_predictions, print_report

load_dotenv()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client[os.getenv("DB_NAME")]

# Collections
logs_collection        = db["agent_logs"]
trends_collection      = db["google_trends"]
predictions_collection = db["predictions"]
analysis_collection    = db["analysis_results"]

# Configure Fireworks via LangChain
llm = ChatFireworks(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    model="accounts/fireworks/models/deepseek-v4-pro",
    temperature=0.7,
    max_tokens=1024
)


# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────

def log(agent, status, message=""):
    timestamp = datetime.now(tz=timezone.utc)
    logs_collection.insert_one({
        "agent":     agent,
        "status":    status,
        "message":   message,
        "timestamp": timestamp
    })
    print(f"[{timestamp.strftime('%H:%M:%S')}] [{agent}] {status} {message}")


# ─────────────────────────────────────────
# SPIKE DETECTION
# ─────────────────────────────────────────

def detect_spikes():
    print("\n🔎 Checking for trend spikes...\n")
    spikes = []

    trends = list(trends_collection.find({}))

    for trend in trends:
        keyword      = trend.get("keyword", "")
        avg_interest = trend.get("avg_interest", 0)
        max_interest = trend.get("max_interest", 0)
        spike_delta  = max_interest - avg_interest

        if spike_delta >= 30:
            spikes.append({
                "keyword":     keyword,
                "avg":         avg_interest,
                "peak":        max_interest,
                "spike_delta": spike_delta
            })
            print(f"  🚨 SPIKE: '{keyword}' jumped {spike_delta} points!")

    if not spikes:
        print("  ✅ No spikes. Trends are stable.\n")

    return spikes


# ─────────────────────────────────────────
# SELF REFLECTION
# ─────────────────────────────────────────

def self_reflect():
    print("\n🧠 Running self reflection...\n")

    all_preds = list(predictions_collection.find({}))
    if len(all_preds) < 2:
        print("  ℹ️ Not enough predictions yet. Need at least 2.\n")
        return

    latest   = sorted(all_preds, key=lambda x: x["predicted_at"], reverse=True)[0]
    previous = sorted(all_preds, key=lambda x: x["predicted_at"], reverse=True)[1]

    current_trends = list(trends_collection.find({}, {"_id": 0, "keyword": 1, "avg_interest": 1}))
    trends_text    = "\n".join([
        f"- {t['keyword']}: {t['avg_interest']}/100"
        for t in current_trends
    ])

    reflection_prompt = PromptTemplate(
        input_variables=["previous_prediction", "latest_prediction", "current_trends"],
        template="""
        You are an AI system reflecting on your own meme trend predictions.

        YOUR PREVIOUS PREDICTION:
        {previous_prediction}

        YOUR LATEST PREDICTION:
        {latest_prediction}

        WHAT IS ACTUALLY TRENDING RIGHT NOW:
        {current_trends}

        Reflect and answer:
        1. Which predictions were accurate so far?
        2. Which predictions were wrong and why?
        3. What patterns did you miss?
        4. How should you adjust your next prediction?

        Give yourself an accuracy score out of 10 and explain why.
        Be brutally honest — this helps you improve.
        """
    )

    try:
        chain    = reflection_prompt | llm
        response = chain.invoke({
            "previous_prediction": previous["prediction"][:1000],
            "latest_prediction":   latest["prediction"][:1000],
            "current_trends":      trends_text
        })
        reflection = response.content

        db["reflections"].insert_one({
            "reflection":   reflection,
            "reflected_at": datetime.now(tz=timezone.utc)
        })

        print("🪞 Self Reflection:")
        print("─" * 50)
        print(reflection)
        print("─" * 50)
        print("✅ Reflection saved to MongoDB!\n")

    except Exception as e:
        print(f"⚠️ Reflection error: {e}")
        traceback.print_exc()


# ─────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────

def orchestrator(spikes, last_scrape, last_analyse, last_predict, last_reflect):
    now = time.time()

    orchestrator_prompt = PromptTemplate(
        input_variables=["spikes", "time_since_scrape", "time_since_analyse",
                         "time_since_predict", "time_since_reflect"],
        template="""
        You are the orchestrator of a meme trend forecasting AI system.
        You decide what the system should do next based on its current state.

        CURRENT SYSTEM STATE:
        - Minutes since last scrape:   {time_since_scrape} minutes
        - Minutes since last analysis: {time_since_analyse} minutes
        - Minutes since last predict:  {time_since_predict} minutes
        - Minutes since last reflect:  {time_since_reflect} minutes
        - Active trend spikes: {spikes}

        YOUR RULES:
        - Scrape if more than 30 minutes since last scrape
        - Analyse if more than 60 minutes since last analysis
        - Predict if more than 120 minutes since last prediction
        - Reflect if more than 180 minutes since last reflection
        - If there are spikes immediately trigger all agents regardless of timing
        - If trends are volatile reduce all intervals by half
        - If trends are stable you can extend intervals

        Respond with ONLY a JSON object, nothing else:
        {{
            "should_scrape":  true or false,
            "should_analyse": true or false,
            "should_predict": true or false,
            "should_reflect": true or false,
            "reason": "one sentence explaining your decision"
        }}
        """
    )

    time_since_scrape  = round((now - last_scrape)  / 60, 1)
    time_since_analyse = round((now - last_analyse) / 60, 1)
    time_since_predict = round((now - last_predict) / 60, 1)
    time_since_reflect = round((now - last_reflect) / 60, 1)
    spikes_text        = str(spikes) if spikes else "none"

    try:
        chain    = orchestrator_prompt | llm
        response = chain.invoke({
            "spikes":              spikes_text,
            "time_since_scrape":   time_since_scrape,
            "time_since_analyse":  time_since_analyse,
            "time_since_predict":  time_since_predict,
            "time_since_reflect":  time_since_reflect,
        })

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        decision = json.loads(raw.strip())
        print(f"\n🤖 Orchestrator: {decision.get('reason', '')}")
        return decision

    except Exception as e:
        print(f"⚠️ Orchestrator error: {e} — using time-based fallback")
        return {
            "should_scrape":  (now - last_scrape)  >= 30 * 60,
            "should_analyse": (now - last_analyse) >= 60 * 60,
            "should_predict": (now - last_predict) >= 120 * 60,
            "should_reflect": (now - last_reflect) >= 180 * 60,
            "reason":         "fallback time-based decision"
        }


# ─────────────────────────────────────────
# AGENT RUNNERS
# ─────────────────────────────────────────

def run_scraper():
    try:
        log("SCRAPER", "▶ STARTING")
        scrape()
        log("SCRAPER", "✅ COMPLETE")
        return time.time()
    except Exception as e:
        log("SCRAPER", "❌ FAILED", str(e))
        return None

def run_analyst():
    try:
        log("ANALYST", "▶ STARTING")
        analyse_trends()
        analyse_memes()
        log("ANALYST", "✅ COMPLETE")
        return time.time()
    except Exception as e:
        log("ANALYST", "❌ FAILED", str(e))
        return None

def run_predictor():
    try:
        log("PREDICTOR", "▶ STARTING")
        trends_analysis, memes_analysis = load_analyses()
        if trends_analysis and memes_analysis:
            prediction = generate_predictions(trends_analysis, memes_analysis)
            if prediction:
                print_report(prediction)
                log("PREDICTOR", "✅ COMPLETE")
                return time.time()
        log("PREDICTOR", "⚠️ SKIPPED", "No analysis data")
        return None
    except Exception as e:
        log("PREDICTOR", "❌ FAILED", str(e))
        return None

def run_reflector():
    try:
        log("REFLECTOR", "▶ STARTING")
        self_reflect()
        log("REFLECTOR", "✅ COMPLETE")
        return time.time()
    except Exception as e:
        log("REFLECTOR", "❌ FAILED", str(e))
        return None


# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("   🤖 MEME TREND FORECASTER — AGENTIC SYSTEM")
    print("   Press Ctrl+C at any time to stop")
    print("=" * 60 + "\n")

    log("SYSTEM", "🚀 STARTED", "Agentic system is now running")

    last_scrape  = 0
    last_analyse = 0
    last_predict = 0
    last_reflect = 0

    while True:
        print("\n" + "─" * 60)
        print(f"⏰ Loop tick at {datetime.now().strftime('%H:%M:%S')}")
        print("─" * 60)

        spikes = detect_spikes()

        decision = orchestrator(
            spikes,
            last_scrape,
            last_analyse,
            last_predict,
            last_reflect
        )

        if decision.get("should_scrape"):
            result = run_scraper()
            if result:
                last_scrape = result

        if decision.get("should_analyse"):
            result = run_analyst()
            if result:
                last_analyse = result

        if decision.get("should_predict"):
            result = run_predictor()
            if result:
                last_predict = result

        if decision.get("should_reflect"):
            result = run_reflector()
            if result:
                last_reflect = result

        print(f"\n⏳ Sleeping 60 seconds before next check...")
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ System stopped by user.")
        log("SYSTEM", "⛔ STOPPED", "Manually stopped by user")