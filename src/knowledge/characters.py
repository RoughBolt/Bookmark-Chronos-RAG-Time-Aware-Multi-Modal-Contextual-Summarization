# Heuristic registry for Identity Aliasing (Edge Case #6 Fix)
MASTER_ALIAS_DICT = {
    "Waymar Royce": ["Ser Waymar", "Ser Waymar Royce", "Waymar Royce", "Royce", "lordling"],
    "Will": ["Will"],
    "Gared": ["Gared"],
    "Mance Rayder": ["Mance Rayder", "Mance"]
}

def extract_characters(text):
    """
    Scans a memory string and returns a set of known master entity names.
    Uses an alias dictionary to normalize variants (e.g. 'lordling' -> 'Waymar Royce').
    """
    found = set()
    t = text.lower()
    
    for master_key, aliases in MASTER_ALIAS_DICT.items():
        for alias in aliases:
            # We add spaces around alias checks or just do simple substring for now
            if alias.lower() in t:
                found.add(master_key)
                break  # Once we find one alias for this master, no need to check others
                
    return list(found)
