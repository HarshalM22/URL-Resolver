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
    hospitals = db.fetch_unprocessed_hospitals(limit=10)

    if not hospitals:
        logger.info("No unprocessed hospitals found.")
        print("no unprocessed hospitals found.")
        return

    # results = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        if "hospital_name" not in reader.fieldnames or "state" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: hospital_name,state")

        for row in reader:
            hospital = row["hospital_name"].strip()
            state = row["state"].strip()

            logger.info(f"Resolving: {hospital}, {state}")

            try:
                resolved = resolver.resolve(hospital, state)

                results.append({
                    "hospital_name": hospital,
                    "state": state,
                    "domain": resolved["domain"],
                    "ownership": resolved["ownership"],
                    "confidence": resolved["confidence"],
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"Failed: {hospital}, {state} | {e}")

                results.append({
                    "hospital_name": hospital,
                    "state": state,
                    "domain": "",
                    "ownership": "",
                    "confidence": "",
                    "status": "failed"
                })

            # polite delay to avoid SERP rate limits
            time.sleep(1)

    write_results(results)


def write_results(rows):
    fieldnames = [
        "hospital_name",
        "state",
        "domain",
        "ownership",
        "confidence",
        "status"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Batch complete. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_batch()
