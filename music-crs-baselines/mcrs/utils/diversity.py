"""Global and session-level diversity for catalog coverage."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple


def apply_score_penalties(
    scored_candidates: Sequence[Tuple[str, float]],
    global_counts: Dict[str, int],
    *,
    global_penalty: float = 2.0,
    max_global_allowed: int = 3,
) -> List[Tuple[str, float]]:
    """Penalize tracks that already appear often across the submission file."""
    adjusted: List[Tuple[str, float]] = []
    for track_id, raw_score in scored_candidates:
        times_used = global_counts.get(track_id, 0)
        if times_used >= max_global_allowed:
            continue
        adjusted.append((track_id, raw_score - (times_used * global_penalty)))
    adjusted.sort(key=lambda item: item[1], reverse=True)
    return adjusted


def select_diverse_topk(
    scored_candidates: Sequence[Tuple[str, float]],
    session_used: Set[str],
    global_counts: Dict[str, int],
    *,
    topk: int = 20,
    global_penalty: float = 2.0,
    max_global_allowed: int = 3,
) -> List[str]:
    """Pick top-k tracks with global frequency penalty and within-session de-duplication."""
    penalized = apply_score_penalties(
        scored_candidates,
        global_counts,
        global_penalty=global_penalty,
        max_global_allowed=max_global_allowed,
    )

    final: List[str] = []
    seen: Set[str] = set()
    for track_id, _ in penalized:
        if track_id in session_used or track_id in seen:
            continue
        final.append(track_id)
        seen.add(track_id)
        if len(final) >= topk:
            break

    if len(final) < topk:
        for track_id, _ in scored_candidates:
            if track_id in seen or track_id in session_used:
                continue
            if global_counts.get(track_id, 0) >= max_global_allowed:
                continue
            final.append(track_id)
            seen.add(track_id)
            if len(final) >= topk:
                break

    session_used.update(final)
    for track_id in final:
        global_counts[track_id] = global_counts.get(track_id, 0) + 1
    return final[:topk]
