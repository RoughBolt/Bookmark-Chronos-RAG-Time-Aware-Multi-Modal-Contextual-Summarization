from src.knowledge.characters import extract_characters

def update_health_status(event_dict, current_state):
    """
    Reads an event dictionary. Handles conflict resolution for character health state.
    States are objects: {"status": "DEAD", "confidence": 0.85, "verified": True}
    """
    event_type = event_dict.get("type")
    
    # A learned classifier event is usually verified, rule_fallback might be rumor
    is_verified = event_dict.get("classifier", "rule") == "learned"
    confidence = 1.0 if is_verified else 0.5
    text = event_dict.get("text", "")
    
    chars_present = []
    if event_type in ["death", "resurrection"]:
        chars_present = extract_characters(text)
        
    for char in chars_present:
        existing_state = current_state.get(char, {"status": "ALIVE", "confidence": 1.0, "verified": True})
        
        if event_type == "death":
            new_status = "DEAD"
        elif event_type == "resurrection":
            new_status = "UNDEAD/ALIVE"
        else:
            continue
            
        # State Rollback & Conflict Resolution
        if not is_verified and existing_state.get("verified", True):
            # Ignore unverified rumor if we have a verified state
            continue
            
        if is_verified and not existing_state.get("verified", True):
            # Verified state overrides rumor completely
            current_state[char] = {"status": new_status, "confidence": confidence, "verified": True}
        else:
            # Same verification level, take the new chronological state
            current_state[char] = {"status": new_status, "confidence": confidence, "verified": is_verified}
            
    return current_state
