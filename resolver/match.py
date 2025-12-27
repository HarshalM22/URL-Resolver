import re
import json 
from utils.logger import logger


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
