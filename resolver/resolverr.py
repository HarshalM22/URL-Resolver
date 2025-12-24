import json
from serp.serp_client import SerpClient
from resolver.canonicalizer import canonical_domain
from resolver.filters import is_blocked
from resolver.ollama_reasoner import call_ollama
from resolver.confidence import compute_final_confidence
from utils.logger import logger
from config.settings import CONFIDENCE_THRESHOLD
from resolver.cms_client import CMSClient
from resolver.match import select_matching_cms_record 




class HospitalDomainResolver:

    def resolve(self, hospital_name: str, state: str):
        queries = [
            f"{hospital_name} {state} hospital official website",
            f"{hospital_name} {state} health system"
        ]

        serp_results = []
        for q in queries:
            serp_results.extend(SerpClient.search(q))


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
        

        candidates = list(domain_map.keys())
        
        if len(candidates) > 3:
            candidates = candidates[:3]

        # candidates = sorted(
        #         domain_map.keys(),
        #         key=lambda d: domain_map[d][0]["rank"]
        #     )[:3]    
        
        if not candidates:
            raise RuntimeError("No valid domains found")
        


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
        
        client = CMSClient()
        content = client.fetch(selected)

        print(f"CMS content for {selected}")
        cms_record = None
        if content:
            cms_record = select_matching_cms_record(hospital_name, content)

            if not cms_record:
                logger.warning(
                    f"No matching CMS record found for {hospital_name} on {selected}"
                )
        else:
            logger.warning(f"No CMS content found for {selected}")


        return {
            "hospital": hospital_name,
            "state": state,
            "domain": selected,
            "ownership": ownership,
            "confidence": round(final_conf, 3),
            "cms": {
                "matched": bool(cms_record),
                "location_name": cms_record.get("location-name") if cms_record else None,
                "mrf_url": cms_record.get("mrf-url") if cms_record else "Not Found",
                "source_page": cms_record.get("source-page-url") if cms_record else None,
                "contact_phone": cms_record.get("contact-phone") if cms_record else None,
                "contact_email": cms_record.get("contact-email") if cms_record else None 
            }
        }
