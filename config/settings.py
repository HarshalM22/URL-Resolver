import os

SERP_API_KEY = os.getenv("SERP_API_KEY")

SERP_ENGINE = "google"  # or "bing"
SERP_RESULTS = 10

OLLAMA_PRIMARY_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_FALLBACK_MODEL = "mistral:7b-instruct-q4_K_M"

OLLAMA_TEMPERATURE = 0.1
OLLAMA_TOP_P = 0.9

CONFIDENCE_THRESHOLD = 0.7


DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": "medcost",
    "port": 3306
}
