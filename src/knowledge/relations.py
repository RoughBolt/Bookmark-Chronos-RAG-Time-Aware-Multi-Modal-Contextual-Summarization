from src.knowledge.characters import extract_characters

def update_interactions(event_dict, interaction_graph):
    """
    Tracks co-occurrence of characters in a single event.
    interaction_graph is a dict: {"Will": {"Gared": 5}, "Waymar Royce": {"Will": 2}}
    """
    chars_present = extract_characters(event_dict.get("text", ""))
    
    # If 2 or more characters are in the same event, they interacted
    for i in range(len(chars_present)):
        for j in range(i + 1, len(chars_present)):
            c1, c2 = chars_present[i], chars_present[j]
            
            if c1 not in interaction_graph: interaction_graph[c1] = {}
            if c2 not in interaction_graph: interaction_graph[c2] = {}
            
            interaction_graph[c1][c2] = interaction_graph[c1].get(c2, 0) + 1
            interaction_graph[c2][c1] = interaction_graph[c2].get(c1, 0) + 1
            
    return interaction_graph
