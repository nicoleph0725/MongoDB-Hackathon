import os
import feedparser
from pytrends.request import TrendReq
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DB_NAME")]

# Collections
trends_collection = db["google_trends"]
memes_collection  = db["meme_posts"]


# ─────────────────────────────────────────
# PART 1 — Google Trends
# ─────────────────────────────────────────

def scrape_google_trends():
    print("📈 Scraping Google Trends...")

    pytrends = TrendReq(hl='en-US', tz=0)

    keywords = [
        "meme", "brainrot", "rizz", "skit", "viral video",
        "NPC", "sigma", "gyatt", "skibidi", "delulu"
    ]

    batches = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
    total   = 0

    for batch in batches:
        try:
            pytrends.build_payload(batch, timeframe='now 1-d', geo='')
            data = pytrends.interest_over_time()

            if data.empty:
                continue

            for keyword in batch:
                if keyword not in data.columns:
                    continue

                avg_interest = int(data[keyword].mean())
                max_interest = int(data[keyword].max())

                record = {
                    "keyword":      keyword,
                    "avg_interest": avg_interest,
                    "max_interest": max_interest,
                    "scraped_at":   datetime.now(tz=timezone.utc),
                    "source":       "google_trends"
                }

                trends_collection.update_one(
                    {"keyword": keyword},
                    {"$set": record},
                    upsert=True
                )
                total += 1
                print(f"  ✅ {keyword}: avg interest {avg_interest}/100")

        except Exception as e:
            print(f"  ⚠️ Error fetching batch {batch}: {e}")

    print(f"\n✅ Google Trends done! {total} keywords tracked.\n")


# ─────────────────────────────────────────
# PART 2 — Know Your Meme RSS
# ─────────────────────────────────────────

RSS_FEEDS = {
    "knowyourmeme_new":     "https://knowyourmeme.com/memes/new.rss",
    "knowyourmeme_popular": "https://knowyourmeme.com/memes.rss",
}

def scrape_rss_feeds():
    print("📰 Scraping Know Your Meme RSS feeds...")
    total = 0

    for source_name, url in RSS_FEEDS.items():
        print(f"\n  🔗 Fetching {source_name}...")
        feed = feedparser.parse(url)

        if not feed.entries:
            print(f"  ⚠️ No entries found for {source_name}")
            continue

        for entry in feed.entries:
            if memes_collection.find_one({"url": entry.link}):
                continue

            record = {
                "title":      entry.get("title", ""),
                "url":        entry.get("link", ""),
                "summary":    entry.get("summary", "")[:500],
                "source":     source_name,
                "scraped_at": datetime.now(tz=timezone.utc),
            }

            memes_collection.insert_one(record)
            total += 1
            print(f"  ✅ Saved: {entry.get('title', '')[:60]}")

    print(f"\n✅ RSS scraping done! {total} new memes saved.\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def scrape():
    print("🚀 Starting Meme Trend Scraper...\n")
    scrape_google_trends()
    scrape_rss_feeds()
    print("🎉 All scraping complete!")

if __name__ == "__main__":
    scrape()