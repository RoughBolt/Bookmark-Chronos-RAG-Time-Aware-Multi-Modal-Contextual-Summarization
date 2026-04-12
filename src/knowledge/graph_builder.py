import json
import os
from src.knowledge.facts import update_health_status
from src.knowledge.relations import update_interactions

def build_global_knowledge_graph(all_events, output_path="data/knowledge.json"):
    """
    Runs sequentially over the chronological timeline of all events to build the factual state.
    """
    global_state = {
        "statuses": {},       # e.g., {"Will": "ALIVE"}
        "interactions": {}    # e.g., {"Will": {"Gared": 5}}
    }
    
    for event in all_events:
        global_state["statuses"] = update_health_status(event, global_state["statuses"])
        global_state["interactions"] = update_interactions(event, global_state["interactions"])
        
    # Default anyone not marked dead to ALIVE if they interacted
    for char in global_state["interactions"].keys():
        if char not in global_state["statuses"]:
            global_state["statuses"][char] = "ALIVE"
            
    # Serialize to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(global_state, f, indent=4)
        
    return global_state
