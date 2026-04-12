import json
import os
from src.knowledge.characters import extract_characters

def generate_hybrid_context(recalled_results, knowledge_file_path="data/knowledge.json"):
    """
    Reads the specific subset of recalled memories, determines the actors present,
    and returns a factual preamble based on the global Knowledge Graph.
    """
    if not os.path.exists(knowledge_file_path):
        return ""

    with open(knowledge_file_path, "r", encoding="utf-8") as f:
        global_kg = json.load(f)

    # 1. Who is in this specific memory chunk?
    local_chars = set()
    for r in recalled_results:
        chars = extract_characters(r["text"])
        local_chars.update(chars)

    if not local_chars:
        return ""

    # 2. Extract facts about ONLY these local characters
    statuses = []
    for char in local_chars:
        # Cross reference the global knowledge graph
        status = global_kg["statuses"].get(char, "UNKNOWN")
        statuses.append(f"{char} [{status}]")

    # 3. Format into a Roster Preamble
    roster_str = ", ".join(statuses)
    
    preamble = f"----- FACTUAL ROSTER -----\nActors Present: {roster_str}\n--------------------------"

    return preamble
