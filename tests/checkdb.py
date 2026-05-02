import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DB_NAME")]

collection = db["analysis_results"]

print(f"Total documents: {collection.count_documents({})}")
print("\nAll documents:")
for doc in collection.find({}, {"_id": 0, "type": 1, "analysed_at": 1}):
    print(doc)