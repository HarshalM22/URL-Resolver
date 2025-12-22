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

Rules:
- Choose ONE domain from the provided candidate_domains.
- NEVER invent or modify domains.
- Prefer independent hospital websites.
- Otherwise choose the parent health system website.
- Ignore directories, social media, and government portals.

Output STRICT JSON only:
{
  "selected_domain": "",
  "ownership": "independent | parent_system",
  "confidence": 0.0,
  "reason": ""
}
""".strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from text safely.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError("No JSON object found in Ollama output")
    return json.loads(match.group(0))


def call_ollama(payload: Dict[str, Any]) -> Dict[str, Any]:
    # ---------- MINIMIZE PAYLOAD ----------
    safe_payload = {
        "hospital_name": payload.get("hospital_name"),
        "state": payload.get("state"),
        "candidate_domains": payload.get("candidate_domains")[:3],
        "evidence": [
            {
                "domain": e["domain"],
                "rank": e.get("top_rank"),
                "summary": e.get("summary", "")[:150]
            }
            for e in payload.get("serp_context", [])[:2]
        ]
    }

    prompt = f"""
SYSTEM:
{SYSTEM_PROMPT}

USER:
{json.dumps(safe_payload, indent=2)}
""".strip()

    try:
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

        raw = response.json().get("response")

        if raw is None:
            raise RuntimeError("Empty response from Ollama")

        # ---------- NORMALIZATION LAYER ----------
        if isinstance(raw, dict):
            parsed = raw

        elif isinstance(raw, str):
            raw = raw.strip()

            # Case 1: pure JSON
            if raw.startswith("{"):
                parsed = json.loads(raw)
            else:
                # Case 2: text + JSON
                parsed = _extract_json(raw)

        else:
            raise RuntimeError(f"Unexpected Ollama output type: {type(raw)}")

        # ---------- VALIDATION ----------
        required = {"selected_domain", "ownership", "confidence", "reason"}
        if not required.issubset(parsed):
            raise RuntimeError("Missing required fields in Ollama output")

        if parsed["selected_domain"] not in safe_payload["candidate_domains"]:
            raise RuntimeError("Selected domain not in candidate list")

        parsed["confidence"] = float(parsed["confidence"])

        if not (0.0 <= parsed["confidence"] <= 1.0):
            raise RuntimeError("Confidence out of range")

        return parsed

    except requests.Timeout:
        raise RuntimeError("Ollama timeout")

    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Ollama returned invalid JSON: {e}")

    except requests.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama")
