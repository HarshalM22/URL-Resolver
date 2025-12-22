import requests
from config.settings import SERP_API_KEY, SERP_ENGINE, SERP_RESULTS

class SerpClient:
    BASE_URL = "https://serpapi.com/search.json"

    @staticmethod
    def search(query: str):
        params = {
            "q": query,
            "engine": SERP_ENGINE,
            "api_key": SERP_API_KEY,
            "num": SERP_RESULTS
        }
        resp = requests.get(SerpClient.BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("organic_results", [])
