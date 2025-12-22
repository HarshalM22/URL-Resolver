import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import csv
import time
from resolver.resolverr import HospitalDomainResolver
from utils.logger import logger
from db.mysql_client import MySQLClient



# INPUT_CSV = "input.csv"
# OUTPUT_CSV = "result.csv"


def run_batch():
    
    db = MySQLClient()

    resolver = HospitalDomainResolver()
    hospitals = db.fetch_unprocessed_hospitals(limit=2)


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

        except Exception as e:
            logger.error(f"Failed: {name} | {e}")

            db.save_result(
                id=hid,
                website=result["domain"],
                websiteURL_ownership=result["ownership"],
                website_confidence=result["confidence"],
            )

        time.sleep(1)

    # write_results(results)


# def write_results(rows):
#     fieldnames = [
#         "hospital_name",
#         "state",
#         "domain",
#         "ownership",
#         "confidence",
#         "status"
#     ]

#     with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
#         writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)

    logger.info(f"Batch complete. Results written")


if __name__ == "__main__":
    run_batch()
