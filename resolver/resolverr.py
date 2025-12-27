import json
from serp.serp_client import SerpClient
from resolver.canonicalizer import canonical_domain
from resolver.filters import is_blocked
from resolver.ollama_reasoner import call_ollama
from resolver.confidence import compute_final_confidence
from utils.logger import logger
from config.settings import CONFIDENCE_THRESHOLD
from resolver.cms_client import CMSClient
from resolver.match import select_matching_cms_record, convert_text_to_json



class HospitalDomainResolver:

    def resolve(self, hospital_name: str, state: str,city:str):
        queries = [
            f"{hospital_name} {city} {state} hospital official website",
            f"{hospital_name} {city} {state} health system"
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
        
        # if len(candidates) > 3:
        #     candidates = candidates[:3]

        candidates = sorted(
                domain_map.keys(),
                key=lambda d: domain_map[d][0]["rank"]
            )[:4]    
        
        if not candidates:
            raise RuntimeError("No valid domains found")
        
        print(f"----------candidates is=============== {candidates}")


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
                for domain, entries in domain_map.items()
                if domain in candidates
            ]
        }


        ai_result = call_ollama(payload,domain_map)

        if ai_result is None:
            print("no response from AI")
            return None

        selected = ai_result["selected_domain"]
        ai_conf = float(ai_result["confidence"])
        ownership = ai_result["ownership"]

        if selected not in domain_map:
            print("AI selected domain is not in domain map")
            return None
        
        serp_rank = domain_map[selected][0]["rank"]
        final_conf = compute_final_confidence(ai_conf, serp_rank, ownership)
        
        client = CMSClient()
        content = client.fetch(selected)

        cms_txt_content = json.dumps(convert_text_to_json(content),indent=4)
        if(not cms_txt_content):
            print(f"JSON content is not available to dump")
        
        cms_record = None
        if content:
            cms_record = select_matching_cms_record(hospital_name, content)
        else:
            print(f"No matching CMS record found for {hospital_name}")

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
                "mrf_url": cms_record.get("mrf-url") if cms_record else  "404 : Not Found",
                "source_page": cms_record.get("source-page-url") if cms_record else  "404 : Not Found",
                "contact_name": cms_record.get("contact-name") if cms_record else  "404 : Not Found",
                "contact_email": cms_record.get("contact-email") if cms_record else  "404 : Not Found" 
            }
        }
