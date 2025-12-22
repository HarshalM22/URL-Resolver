import tldextract

def canonical_domain(url: str) -> str | None:
    if not url:
        return None
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"
