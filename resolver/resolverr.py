import json
from serp.serp_client import SerpClient
from resolver.canonicalizer import canonical_domain
from resolver.filters import is_blocked
from resolver.ollama_reasoner import call_ollama
from resolver.confidence import compute_final_confidence
from utils.logger import logger
from config.settings import CONFIDENCE_THRESHOLD

class HospitalDomainResolver:

    def resolve(self, hospital_name: str, state: str):
        queries = [
            f"{hospital_name} {state} hospital official website",
            f"{hospital_name} {state} health system"
        ]

        serp_results = []
        for q in queries:
            serp_results.extend(SerpClient.search(q))

        print(f"serp results ========================{serp_results}")

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
        
        print(f"domain map ======================= {domain_map}")

        candidates = list(domain_map.keys())
        
        if len(candidates) > 3:
            candidates = candidates[:3]
        
        if not candidates:
            raise RuntimeError("No valid domains found")
        
        print(f"candidates ======================== {candidates}")


        payload = {
            "hospital_name": hospital_name,
            "state": state,
            "candidate_domains": candidates,
            "serp_context": [
                {
                    "domain": domain,
                    "top_rank": entries[0]["rank"],
                    "summary": entries[0]["snippet"][:100]
                }
                for domain, entries in domain_map.items()
                if domain in candidates
            ]
        }

        ai_result = call_ollama(payload)
        # print(f"raw AI response ================================ {raw}")

        # ai_result = json.loads(raw)

        # print(f"AI result ======================================= {ai_result}")

        selected = ai_result["selected_domain"]
        ai_conf = float(ai_result["confidence"])
        ownership = ai_result["ownership"]

        if selected not in candidates or ai_conf < CONFIDENCE_THRESHOLD:
            logger.warning("AI fallback triggered, using top SERP domain")
            selected = candidates[0]
            ownership = "unknown"
            ai_conf = 0.7

        serp_rank = domain_map[selected][0]["rank"]
        final_conf = compute_final_confidence(ai_conf, serp_rank, ownership)

        return {
            "hospital": hospital_name,
            "state": state,
            "domain": selected,
            "ownership": ownership,
            "confidence": round(final_conf, 3)
        }
