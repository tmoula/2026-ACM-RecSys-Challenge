"""Shared conversation parsing for inference scripts."""

from typing import Any, Dict, List, Tuple


def parse_conversation_history(
    conversations: List[Dict[str, Any]],
    music_crs: Any,
    target_turn_number: int,
) -> Tuple[List[Dict[str, str]], str]:
    """Build chat history up to a target turn, expanding music turns to metadata."""
    chat_history: List[Dict[str, str]] = []
    user_query = ""

    for turn in conversations:
        turn_number = turn["turn_number"]
        if turn_number < target_turn_number:
            role = turn["role"]
            content = turn["content"]
            if role == "music":
                role = "assistant"
                content = music_crs.item_db.id_to_metadata(turn["content"])
            chat_history.append({"role": role, "content": content})
        elif turn_number == target_turn_number:
            user_query = turn["content"]

    return chat_history, user_query
