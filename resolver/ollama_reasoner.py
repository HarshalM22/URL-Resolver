import json
import re
import requests
from typing import Dict, Any
from config.settings import (
    OLLAMA_PRIMARY_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P
)

OLLAMA_URL = "http://localhost:11434/api/generate"


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



def call_ollama(payload: Dict[str, Any],domain_map) -> Dict[str, Any]:
    candidate_domains = payload.get("candidate_domains") or []

    safe_payload = {
        "hospital_name": payload.get("hospital_name"),
        "state": payload.get("state"),
        "candidate_domains": candidate_domains,
        "evidence": {
            domain: entries[:4]
            for domain, entries in domain_map.items()
            if domain in candidate_domains
            }
,
    }

    prompt = f"""
SYSTEM:
{SYSTEM_PROMPT}

USER:
{json.dumps(safe_payload)}
""".strip()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_PRIMARY_MODEL,
            "prompt": prompt,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    # print("json reponse is ", json.loads(response.json()))
    return json.loads(response.json()["response"])

    # raw_response = response.json().get("response")

    # if not raw_response:
    #     raise RuntimeError("Empty response from Ollama")

    # Always try strict path first
    # try:
    #     parsed = json.loads(raw_response)

    #     # ---- STRICT VALIDATION ----
    #     required = {"selected_domain", "ownership", "confidence", "reason"}
    #     if not required.issubset(parsed):
    #         raise ValueError("Missing required fields")

    #     if parsed["selected_domain"] not in candidate_domains:
    #         raise ValueError("Selected domain not in candidate list")

    #     parsed["confidence"] = float(parsed["confidence"])
    #     if not (0.0 <= parsed["confidence"] <= 1.0):
    #         raise ValueError("Confidence out of range")

    #     # ✅ VALID RESULT
    #     parsed["_status"] = "validated"
    #     return parsed

    # except Exception as validation_error:
    #     # ⚠️ FALLBACK PATH — DO NOT LOSE DATA
    #     return {
    #         "_status": "unvalidated",
    #         "_error": str(validation_error),
    #         "_raw_model_response": raw_response
    #     }
