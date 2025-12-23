import re

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
    Select the CMS record that best matches the hospital name.
    """

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

        # Exact or substring match → immediate win
        if target in loc_norm or loc_norm in target:
            return record

        # Token overlap score
        overlap = len(target_tokens & loc_tokens)

        if overlap > best_score:
            best_score = overlap
            best_match = record

    # Confidence threshold (important)
    if best_score >= max(2, len(target_tokens) // 2):
        return best_match

    return None
