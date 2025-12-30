# ============================================================
# STANDARD LIBS
# ============================================================
import sys
import os
import json
import csv
import time
import re
import logging
from typing import Dict, Any, Optional, List
import pymysql
import random
import requests
import httpx
# ============================================================
# THIRD-PARTY LIBS
# ============================================================
# from curl_cffi import requests as async_requests
import mysql.connector
import tldextract
from dotenv import load_dotenv
from etl_app.config import Settings
from etl_app.db import engine 
from sqlalchemy import text
import asyncio
from curl_cffi.requests import AsyncSession

# ============================================================
# LOGGER (utils/logger.py)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("hospital-resolver")

# ============================================================
# DB CLIENT (db/mysql_client.py)
# ============================================================
class MySQLClient:
    def __init__(self):
        self.engine = engine


    def fetch_unprocessed_hospitals(self, limit):
        with self.engine.begin() as conn:
            query = text("""
                SELECT id, name, state, city
                FROM hospitals
                WHERE website IS NULL OR website = ''
                LIMIT :limit
            """)
            result = conn.execute(query, {"limit": limit})
            return result.mappings().all()
        

    def save_result(
        self,
        id,
        website,
        websiteURL_ownership,
        website_confidence,
        cms_txt_content 
    ):
        with self.engine.begin() as conn:
            query = text("""
                UPDATE hospitals
                SET website = :website,
                    websiteURL_ownership = :ownership,
                    website_confidence = :confidence,
                    cms_txt_content = :content
                WHERE id = :id
            """)
            conn.execute(query, {
                "website": website,
                "ownership": websiteURL_ownership,
                "confidence": website_confidence,
                "content": cms_txt_content,
                "id": id
            })

    
    def Save_mrfResult(self,hid,mrf_link,meta ):
        
        with self.engine.begin() as conn:
            query = text("""
                INSERT INTO hospital_mrf_links (hospital_id, mrf_url, discovered_by, meta)
                VALUES (:hid, :mrf_link, 'cms-hpt', :meta)
            """)
            conn.execute(query, {
                "hid": hid,
                "mrf_link": mrf_link,
                "meta": meta
            })
# ============================================================
# SERP CLIENT (serp/serpapi_client.py)
# ============================================================
class SerpClient:
    BASE_URL = "https://serpapi.com/search.json"

    @staticmethod
    def search(query: str):
        params = {
            "q": query,
            "engine": Settings.SERP_ENGINE,
            "api_key": Settings.SERP_API_KEY,
            "num": Settings.SERP_RESULTS
        }
        resp = requests.get(SerpClient.BASE_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("organic_results", [])

# ============================================================
# FILTERS (resolver/filter.py)
# ============================================================
BLOCKLIST = {
    "wikipedia.org", "facebook.com", "linkedin.com", "healthgrades.com",
    "webmd.com", "yelp.com", "mapquest.com", "indeed.com", "glassdoor.com"
}

def is_blocked(domain: str) -> bool:
    return domain in BLOCKLIST

# ============================================================
# DOMAIN CANONICALIZER (resolver/canonicalizer.py)
# ============================================================
def canonical_domain(url: str) -> str | None:
    if not url:
        return None
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"

# ============================================================
# CONFIDENCE (resolver/confidence.py)
# ============================================================
def compute_final_confidence(
    ai_confidence: float,
    serp_rank: int,
    ownership: str
) -> float:
    score = ai_confidence

    if serp_rank == 1:
        score += 0.05
    elif serp_rank > 3:
        score -= 0.05

    if ownership == "parent_system":
        score -= 0.05

    return max(0.0, min(1.0, score))

# ============================================================
# CMS CLIENT (resolver/cms_client.py)
# ============================================================
def parse_cms_hpt_records(content: str) -> list[dict]:
    records = []
    current = {}

    for line in content.splitlines():
        line = line.strip()

        if not line:
            if current:
                records.append(current)
                current = {}
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()

    if current:
        records.append(current)

    return records

class CMSClient:
    @staticmethod

    async def fetch(domain: str, timeout: int = 100) -> List[Dict[str, Any]]:
        """
        Enhanced Fetcher to bypass 403/404 blocks.
        Matches your pipeline by returning a list of parsed records.
        """
        browser_targets = ["chrome120", "chrome119","chrome", "safari", "safari_ios"]
        target = random.choice(browser_targets)
        
        async with AsyncSession() as session:

            domain = domain.strip().lower().replace("https://", "").replace("http://", "").rstrip('/')
            homepage_url = f"https://{domain}/"
            cms_url = f"https://{domain}/cms-hpt.txt"

            headers = {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }

            try:
                # --- STEP 1: INITIAL WARM-UP (Non-blocking) ---
                logger.info(f"Establishing session: {homepage_url}")
                await session.get(
                    homepage_url, 
                    impersonate=target, 
                    headers=headers, 
                    timeout=15,
                    verify=False  
                )

                # --- STEP 2: HUMAN BEHAVIOR DELAY (Non-blocking) ---
                # This allows other tasks to run while this one waits
                await asyncio.sleep(random.uniform(2.5, 4.5))

                # --- STEP 3: THE TARGETED FETCH ---
                headers["Referer"] = homepage_url
                
                logger.info(f"Fetching CMS file: {cms_url}")
                resp = await session.get(
                    cms_url,
                    impersonate=target,
                    headers=headers,
                    timeout=timeout
                )

                if resp.status_code == 200:
                    content = resp.text
                    # parse_cms_hpt_records remains sync (CPU bound), which is fine here
                    records = parse_cms_hpt_records(content) 
                    
                    if not records:
                        logger.warning(f"File found but no records parsed for {domain}")
                        return []
                    
                    return records

                elif resp.status_code == 403:
                    logger.error(f"403 Forbidden on {domain}. IP might be flagged.")
                elif resp.status_code == 404:
                    logger.warning(f"404 Not Found: Check if file moved for {domain}")
                
            except Exception as e:
                logger.error(f"Critical Fetch Error for {domain}: {e}")

            return []


# ============================================================
# CMS MATCHING (resolver/match.py)
# ============================================================
def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def select_matching_cms_record(
    hospital_name: str,
    cms_records: list[dict]
) -> dict | None:
    """
    Select best CMS record.
    - Strong match → return it
    - Single record → safe fallback
    - Multiple weak records → return None
    """

    if not cms_records:
        return None

    # SINGLE RECORD SAFETY (IMPORTANT)
    if len(cms_records) == 1:
        logger.info(
            "Single CMS record found — accepting as fallback match"
        )
        return cms_records[0]

    target = normalize_name(hospital_name)
    target_tokens = set(target.split())

    best_match = None
    best_score = 0

    for record in cms_records:
        location = record.get("location-name")
        if not location:
            continue

        loc_norm = normalize_name(location)
        loc_tokens = set(loc_norm.split())

        # Strong substring match
        if target in loc_norm or loc_norm in target:
            return record

        overlap = len(target_tokens & loc_tokens)

        if overlap > best_score:
            best_score = overlap
            best_match = record

    # Threshold-based acceptance
    if best_score >= max(2, len(target_tokens) // 2):
        return best_match

    logger.warning(
        "CMS records exist but no strong identity match found"
    )
    return None


def convert_text_to_json(raw_text):
    # If CMSClient already parsed it, return directly
    if isinstance(raw_text, list):
        return raw_text

    if not isinstance(raw_text, str):
        raise TypeError(f"Expected str or list, got {type(raw_text)}")

    lines = [
        line.strip()
        for line in raw_text.split("\n")
        if line.strip()
    ]

    results = []
    current_entry = {}

    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()

            if key == "location-name":
                if current_entry:
                    results.append(current_entry)
                current_entry = {key: value}
            else:
                current_entry[key] = value

    if current_entry:
        results.append(current_entry)

    return results

# ============================================================
# OLLAMA REASONER (resolver/ollama_reasoner.py)
# ============================================================
SYSTEM_PROMPT = """
You are a hospital website authority resolution engine.

Your task is to select the SINGLE most accurate and authoritative official website
for the given hospital from the provided candidate_domains.

You MUST follow these rules strictly and in the exact priority order below.

SELECTION PRIORITY (MOST IMPORTANT FIRST):

1. IDENTITY MATCH (CRITICAL)
   - The selected domain MUST clearly and specifically represent the given hospital.
   - The hospital name and location (city/state) must strongly match the domain’s
     branding, title, or description.
   - If an independent-looking domain does NOT clearly match the hospital identity,
     it MUST be rejected.

2. OWNERSHIP PREFERENCE
   - Prefer an independent hospital website ONLY IF the identity match is strong.
   - If independent options are weak, ambiguous, or inaccurate, select the official
     parent health system website instead.

3. EXCLUSIONS (ABSOLUTE)
   - NEVER select government portals (e.g., medicare.gov, cms.gov).
   - NEVER select directories, aggregators, associations, or listings.
   - NEVER select social media, news, or third-party informational sites.

4. DOMAIN INTEGRITY
   - You may ONLY choose from the provided candidate_domains.
   - NEVER invent, modify, normalize, or infer domains not present in the list.

5. TIE-BREAKING
   - If multiple domains appear valid, choose the one that:
     a) Most closely matches the hospital name and location
     b) Is least generic
     c) Represents the most direct authority

OWNERSHIP CLASSIFICATION:
- Use "independent" ONLY when the hospital operates under its own clearly branded domain.
- Use "parent_system" when the hospital is owned or operated by a larger health system.

CONFIDENCE SCORING GUIDELINES:
- 0.95–1.0 → Exact hospital match with clear authority
- 0.80–0.94 → Strong parent system match
- 0.60–0.79 → Acceptable but some ambiguity
- ≤0.59 → Weak match; only select if no better option exists

STRICT OUTPUT REQUIREMENTS:
- Output MUST be valid JSON only.
- DO NOT include markdown, explanations, or extra text.
- DO NOT include trailing commas.
- DO NOT include additional fields.

OUTPUT FORMAT (EXACT):
{
  "selected_domain": "<one domain from candidate_domains>",
  "ownership": "independent | parent_system",
  "confidence": 0.0,
  "reason": "<brief factual justification>"
}

FAILSAFE RULE:
- If all independent domains are weak or inaccurate, you MUST select the parent
  health system website, even if independent options exist.


""".strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from text safely.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError("No JSON object found in Ollama output")
    return json.loads(match.group(0))



# def call_ollama(payload: Dict[str, Any],domain_map) -> Dict[str, Any]:
#     candidate_domains = payload.get("candidate_domains") or []

#     safe_payload = {
#         "hospital_name": payload.get("hospital_name"),
#         "state": payload.get("state"),
#         "candidate_domains": candidate_domains,
#         "evidence": {
#             domain: entries[:4]
#             for domain, entries in domain_map.items()
#             if domain in candidate_domains
#             }
# ,
#     }

#     prompt = f"""
# SYSTEM:
# {SYSTEM_PROMPT}

# USER:
# {json.dumps(safe_payload)}
# """.strip()

#     response = requests.post(
#         Settings.OLLAMA_URL,
#         json={
#             "model": Settings.OLLAMA_PRIMARY_MODEL,
#             "prompt": prompt,
#             "temperature": Settings.OLLAMA_TEMPERATURE,
#             "top_p": Settings.OLLAMA_TOP_P,
#             "stream": False
#         },
#         timeout=120
#     )
#     response.raise_for_status()
#     # print("json reponse is ", json.loads(response.json()))
#     return json.loads(response.json()["response"])

async def call_ollama_async(payload: Dict[str, Any], domain_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronous wrapper for Ollama. 
    Uses httpx to prevent blocking the event loop during generation.
    """
    
    # 1. LOGIC PRESERVED: Payload Construction
    candidate_domains = payload.get("candidate_domains") or []

    safe_payload = {
        "hospital_name": payload.get("hospital_name"),
        "state": payload.get("state"),
        "candidate_domains": candidate_domains,
        "evidence": {
            domain: entries[:4]
            for domain, entries in domain_map.items()
            if domain in candidate_domains
        },
    }

    # 2. LOGIC PRESERVED: Prompt Construction
    prompt = f"""
SYSTEM:
{SYSTEM_PROMPT}

USER:
{json.dumps(safe_payload)}
""".strip()

    # 3. ASYNC TRANSFORMATION: Network Request
    # We use a timeout of 120s 
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                Settings.OLLAMA_URL,
                json={
                    "model": Settings.OLLAMA_PRIMARY_MODEL,
                    "prompt": prompt,
                    "temperature": Settings.OLLAMA_TEMPERATURE,
                    "top_p": Settings.OLLAMA_TOP_P,
                    "stream": False,
                    "format": "json" # Highly recommended to force JSON mode 
                }
            )
            
            response.raise_for_status()
            
            # 4. LOGIC PRESERVED: Parsing
            # Ollama returns the generated text in the 'response' field
            return json.loads(response.json()["response"])

        except httpx.TimeoutException:
            # Handles the case where local GPU is overloaded by 500 requests
            print(f"Ollama Timed Out for {payload.get('hospital_name')}")
            return None
        except Exception as e:
            print(f"Ollama Async Error: {e}")
            return None

# ============================================================
# RESOLVER (resolver/resolverr.py)
# ============================================================
class HospitalDomainResolver:

    async def resolve_async(self, hospital_name: str, state: str, city: str):
        """
        Asynchronous version of the hospital domain resolver.
        """
        queries = [
            f"{hospital_name} {city} {state} hospital official website",
            f"{hospital_name} {city} {state} health system"
        ]

        # 1. ASYNC DISCOVERY: Search concurrently for all queries
        # Using asyncio.gather here allows both search queries to run at once
        search_tasks = [SerpClient.search_async(q) for q in queries]
        results_lists = await asyncio.gather(*search_tasks)
        
        serp_results = []
        for r_list in results_lists:
            serp_results.extend(r_list)

        # 2. HEURISTIC MAPPING
        domain_map = {}
        for idx, r in enumerate(serp_results):
            domain = canonical_domain(r.get("link"))
            if not domain or is_blocked(domain):
                continue
            domain_map.setdefault(domain, []).append({
                "rank": idx + 1,
                "title": r.get("title"),
                "snippet": r.get("snippet")
            })

        # 3. CANDIDATE SELECTION
        candidates = sorted(
            domain_map.keys(),
            key=lambda d: domain_map[d][0]["rank"]
        )[:4]

        if not candidates:
            logger.warning(f"No valid domains found for {hospital_name}")
            return {"status": "failed", "reason": "no_candidates"}

        payload = {
            "hospital_name": hospital_name,
            "state": state,
            "candidate_domains": candidates,
            "serp_context": [
                {
                    "domain": domain,
                    "top_rank": entries[0]["rank"],
                    "title": entries[0]["title"],
                    "summary": entries[0]["snippet"][:100]
                }
                for domain, entries in domain_map.items() if domain in candidates
            ]
        }

        # 4. ASYNC AI REASONING
        # This prevents the script from freezing while Ollama processes the prompt
        ai_result = await call_ollama_async(payload, domain_map)

        if ai_result is None:
            return {"status": "failed", "reason": "ai_no_response"}

        selected = ai_result["selected_domain"]
        ai_conf = float(ai_result["confidence"])
        ownership = ai_result["ownership"]

        if selected not in domain_map:
            return {"status": "failed", "reason": "domain_mismatch"}

        # 5. ASYNC STEALTH FETCH (Using your new async fetcher)
        serp_rank = domain_map[selected][0]["rank"]
        final_conf = compute_final_confidence(ai_conf, serp_rank, ownership)
        
        client = CMSClient()
        # This now calls the 'await session.get' logic we built
        content = await client.fetch_async(selected)

        # 6. DATA TRANSFORMATION
        cms_txt_content = json.dumps(convert_text_to_json(content), indent=4)
        
        cms_record = None
        if content:
            cms_record = select_matching_cms_record(hospital_name, content)

        return {
            "hospital": hospital_name,
            "status": "resolved",
            "state": state,
            "domain": selected,
            "ownership": ownership,
            "confidence": round(final_conf, 3),
            "cms_txt_content": cms_txt_content if cms_txt_content else "404 : Not Found",
            "cms": {
                "matched": bool(cms_record),
                "location_name": cms_record.get("location-name") if cms_record else "404 :Not Found",
                "mrf_url": cms_record.get("mrf-url") if cms_record else "404 : Not Found",
                "source_page": cms_record.get("source-page-url") if cms_record else "404 : Not Found",
                "contact_name": cms_record.get("contact-name") if cms_record else "404 : Not Found",
                "contact_email": cms_record.get("contact-email") if cms_record else "404 : Not Found" 
            }
        }

    # def resolve(self, hospital_name: str, state: str,city:str):
    #     queries = [
    #         f"{hospital_name} {city} {state} hospital official website",
    #         f"{hospital_name} {city} {state} health system"
    #     ]

    #     serp_results = []
    #     for q in queries:
    #         serp_results.extend(SerpClient.search(q))


    #     domain_map = {}
    #     for idx, r in enumerate(serp_results):
    #         domain = canonical_domain(r.get("link"))
    #         if not domain or is_blocked(domain):
    #             continue
    #         domain_map.setdefault(domain, []).append({
    #             "rank": idx + 1,
    #             "title": r.get("title"),
    #             "snippet": r.get("snippet")
    #         })
        

    #     candidates = list(domain_map.keys())
        
    #     # if len(candidates) > 3:
    #     #     candidates = candidates[:3]

    #     candidates = sorted(
    #             domain_map.keys(),
    #             key=lambda d: domain_map[d][0]["rank"]
    #         )[:4]    
        
    #     if not candidates:
    #         raise RuntimeError("No valid domains found")
        
    #     print(f"----------candidates is=============== {candidates}")


    #     payload = {
    #         "hospital_name": hospital_name,
    #         "state": state,
    #         "candidate_domains": candidates,
    #         "serp_context": [
    #             {
    #                 "domain": domain,
    #                 "top_rank": entries[0]["rank"],
    #                 "title": entries[0]["title"],
    #                 "summary": entries[0]["snippet"][:100]
    #             }
    #             for domain, entries in domain_map.items()
    #             if domain in candidates
    #         ]
    #     }


    #     ai_result = call_ollama(payload,domain_map)

    #     if ai_result is None:
    #         print("no response from AI")
    #         return None

    #     selected = ai_result["selected_domain"]
    #     ai_conf = float(ai_result["confidence"])
    #     ownership = ai_result["ownership"]

    #     if selected not in domain_map:
    #         print("AI selected domain is not in domain map")
    #         return None
        
    #     serp_rank = domain_map[selected][0]["rank"]
    #     final_conf = compute_final_confidence(ai_conf, serp_rank, ownership)
        
    #     client = CMSClient()
    #     content = client.fetch(selected)

    #     cms_txt_content = json.dumps(convert_text_to_json(content),indent=4)
    #     if(not cms_txt_content):
    #         print(f"JSON content is not available to dump")
        
    #     cms_record = None
    #     if content:
    #         cms_record = select_matching_cms_record(hospital_name, content)
    #     else:
    #         print(f"No matching CMS record found for {hospital_name}")

    #     return {
    #         "hospital": hospital_name,
    #         "status": "resolved",
    #         "state": state,
    #         "domain": selected,
    #         "ownership": ownership,
    #         "confidence": round(final_conf, 3),
    #         "cms_txt_content": cms_txt_content if cms_txt_content else "404 : Not Found",
    #         "cms": {
    #             "matched": bool(cms_record),
    #             "location_name": cms_record.get("location-name") if cms_record else "404 :Not Found",
    #             "mrf_url": cms_record.get("mrf-url") if cms_record else  "404 : Not Found",
    #             "source_page": cms_record.get("source-page-url") if cms_record else  "404 : Not Found",
    #             "contact_name": cms_record.get("contact-name") if cms_record else  "404 : Not Found",
    #             "contact_email": cms_record.get("contact-email") if cms_record else  "404 : Not Found" 
    #         }
    #     }

# ============================================================
# BATCH RUNNER (main)
# ============================================================
# def run_batch():

#     # print(Settings.DB_CONFIG.get("password"))
    
#     db = MySQLClient()

#     resolver = HospitalDomainResolver()
    
#     print(f"fetching hospitals from DB")
    
#     hospitals = db.fetch_unprocessed_hospitals(limit=10)

#     print(f"hospitals fetched from DB: {hospitals}")

#     if not hospitals:
#         logger.info("No unprocessed hospitals found.")
#         print("no unprocessed hospitals found.")
#         return
    


#     # results = []

#     for row in hospitals:
#         hid = row["id"]
#         name = row["name"]
#         state = row["state"]
#         city=row["city"]
        

#         logger.info(f"Resolving: {name}, {state},{city}")

#         try:
#             result = resolver.resolve(name, state,city)

#             if result.get("status") != "resolved":
#                 print("something went wrong, result is None")
#             else:    
#                 db.save_result(
#                     id=hid,
#                     website=result["domain"],
#                     websiteURL_ownership=result["ownership"],
#                     website_confidence=result["confidence"],
#                     cms_txt_content=result["cms_txt_content"]
#                 )

#                 cms = result.get("cms", {})

#                 db.Save_mrfResult(
#                     hid=hid,
#                     mrf_link=cms.get("mrf_url"),
#                     meta=json.dumps({
#                         "source_page": cms.get("source_page"),
#                         "contact_phone": cms.get("contact_phone"),
#                         "contact_email": cms.get("contact_email")
#                     })
#                 )

            

#         except Exception as e:
#             logger.error(f"Failed: {name} | {e}")


#         time.sleep(1)


#     logger.info(f"Batch complete. Results written")


# if __name__ == "__main__":
#     run_batch()


# ============================================================
# ASYNC WORKER: Resolves One Hospital
# ============================================================

async def process_single_hospital(row, resolver, db, semaphore):
    """
    A single worker task. The semaphore prevents too many simultaneous 
    connections or AI calls from crashing your 8GB RAM / 4GB GPU.
    """
    async with semaphore:
        hid = row["id"]
        name = row["name"]
        state = row["state"]
        city = row["city"]

        logger.info(f"🚀 Resolving: {name}, {state}, {city}")

        try:
            # IMPORTANT: resolver.resolve must be changed to 'async def'
            # inside your HospitalDomainResolver class.
            result = await resolver.resolve_async(name, state, city)

            if not result or result.get("status") != "resolved":
                logger.warning(f"⚠️ Failed to resolve: {name}")
                return

            # Update DB using async methods
            await db.save_result_async(
                id=hid,
                website=result["domain"],
                websiteURL_ownership=result["ownership"],
                website_confidence=result["confidence"],
                cms_txt_content=result["cms_txt_content"]
            )

            cms = result.get("cms", {})
            await db.Save_mrfResult_async(
                hid=hid,
                mrf_link=cms.get("mrf_url"),
                meta=json.dumps({
                    "source_page": cms.get("source_page"),
                    "contact_phone": cms.get("contact_phone"),
                    "contact_email": cms.get("contact_email")
                })
            )
            logger.info(f"✅ Success: {name}")

        except Exception as e:
            logger.error(f"❌ Critical Error on {name}: {e}")

# ============================================================
# ASYNC MAIN BATCH
# ============================================================
async def run_batch_async(limit=500):
    """
    The main entry point. Fetches data and manages concurrency.
    """
    db = MySQLClient()
    resolver = HospitalDomainResolver()
    
    # 1. Fetch names from DB (Keep this sync for now or async-ify db.fetch)
    logger.info(f"Fetching {limit} hospitals from DB...")
    hospitals = db.fetch_unprocessed_hospitals(limit=limit)

    if not hospitals:
        logger.info("No unprocessed hospitals found.")
        return

    # 2. Concurrency Control (Semaphore)
    # Set this to 50. It allows 500 tasks to 'start', but only 50 to 
    # be actively using your RAM/Network/GPU at one time.
    concurrency_limit = asyncio.Semaphore(50)

    # 3. Create Task List
    tasks = [
        process_single_hospital(row, resolver, db, concurrency_limit) 
        for row in hospitals
    ]

    # 4. Fire them all simultaneously!
    start_time = time.perf_counter()
    logger.info(f"🔥 Launching async batch for {len(tasks)} hospitals...")
    
    await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    logger.info(f"🏁 Batch complete in {end_time - start_time:.2f} seconds.")

# ============================================================
# START ENGINE
# ============================================================
if __name__ == "__main__":
    try:
        asyncio.run(run_batch_async(limit=10))
    except KeyboardInterrupt:
        logger.info("Batch cancelled by user.")