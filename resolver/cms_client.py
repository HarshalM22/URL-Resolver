import requests
import json
from typing import Optional, Dict, Any
from utils.logger import logger

CMS_PATH = "/cms-hpt.txt"


class CMSClient:
    @staticmethod
    def fetch(domain: str, timeout: int = 300) -> Optional[Dict[str, Any]]:
        """
        Fetches and parses CMS hospital price transparency data.

        Returns:
            Parsed JSON dict if successful
            None if not available or invalid
        """

        url = f"https://{domain}{CMS_PATH}"

        logger.info(f"Fetching page content of : {url}")

        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (CMS-Compliance-Checker)"
                }
            )

            if resp.status_code != 200:
                logger.warning(
                    f"CMS file not found ({resp.status_code}) for {domain}"
                )
                return []

            # Decode safely (CMS files often have odd encoding)
            content = resp.content.decode("utf-8", errors="ignore")

            status = resp.status_code

            records = parse_cms_hpt_records(content)

            # print(f"content of .txt in json format =========={records}") 

            if not records:
                logger.warning(f"No records found in CMS file for {domain}")
                return []

            return records

            
            

        except requests.Timeout:
            logger.error(f"Timeout fetching CMS file for {domain}")
            return []

        except (requests.ConnectionError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching CMS file for {domain}: {e}")
            return []



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