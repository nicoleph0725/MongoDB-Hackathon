import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Test MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DB_NAME")]
db.test.insert_one({"hello": "world"})
print("✅ MongoDB connected!")
