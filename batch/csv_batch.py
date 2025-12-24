import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import csv
import time
from resolver.resolverr import HospitalDomainResolver
from utils.logger import logger
from db.mysql_client import MySQLClient

from resolver.cms_client import CMSClient


def run_batch():
    
    db = MySQLClient()

    resolver = HospitalDomainResolver()
    
    print(f"fetching hospitals from DB")
    
    hospitals = db.fetch_unprocessed_hospitals(limit=2)

    print(f"hospitals fetched from DB: {hospitals}")

    if not hospitals:
        logger.info("No unprocessed hospitals found.")
        print("no unprocessed hospitals found.")
        return
    
  
    

    # results = []

    for row in hospitals:
        hid = row["id"]
        name = row["name"]
        state = row["state"]
        

        logger.info(f"Resolving: {name}, {state}")

        try:
            result = resolver.resolve(name, state)


            db.save_result(
                id=hid,
                website=result["domain"],
                websiteURL_ownership=result["ownership"],
                website_confidence=result["confidence"],
            )

            cms = result.get("cms", {})

            db.Save_mrfResult(
                hid=hid,
                mrf_link=cms.get("mrf_url"),
                meta=json.dumps({
                    "source_page": cms.get("source_page"),
                    "contact_phone": cms.get("contact_phone"),
                    "contact_email": cms.get("contact_email")
                })
            )

            

        except Exception as e:
            logger.error(f"Failed: {name} | {e}")


        time.sleep(1)


    logger.info(f"Batch complete. Results written")


if __name__ == "__main__":
    run_batch()
