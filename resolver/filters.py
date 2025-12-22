BLOCKLIST = {
    "wikipedia.org", "facebook.com", "linkedin.com", "healthgrades.com",
    "webmd.com", "yelp.com", "mapquest.com", "indeed.com", "glassdoor.com"
}

def is_blocked(domain: str) -> bool:
    return domain in BLOCKLIST
