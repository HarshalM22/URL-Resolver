def compute_final_confidence(
    ai_confidence: float,
    serp_rank: int,
    ownership: str
) -> float:
    score = ai_confidence

    if serp_rank == 1:
        score += 0.05
    elif serp_rank > 3:
        score -= 0.05

    if ownership == "parent_system":
        score -= 0.05

    return max(0.0, min(1.0, score))
