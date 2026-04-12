from src.knowledge.characters import extract_characters

def update_health_status(event_dict, current_state):
    """
    Reads an event dictionary. If it's a DEATH event, marks the character DEAD.
    Returns the updated global character state.
    """
    # current_state is a dict: {"Waymar Royce": "ALIVE", "Will": "ALIVE"}
    
    if event_dict.get("type") == "death":
        chars_present = extract_characters(event_dict.get("text", ""))
        for char in chars_present:
            current_state[char] = "DEAD"
            
    elif event_dict.get("type") == "resurrection":
        chars_present = extract_characters(event_dict.get("text", ""))
        for char in chars_present:
            if current_state.get(char) == "DEAD":
                current_state[char] = "UNDEAD/ALIVE"
            
    return current_state
