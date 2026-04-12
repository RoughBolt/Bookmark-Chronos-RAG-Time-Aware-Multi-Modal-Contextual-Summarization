# ===========================VERSION 1===========================

# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.tag import pos_tag

# def extract_events(text):
#     sentences = sent_tokenize(text)
#     events = []

#     for sent in sentences:
#         words = word_tokenize(sent)
#         events = []

#         for sent in sentences:
#             words = word_tokenize(sent)
#             tagged = pos_tag(words)

#             has_verb = any(tag.startswith('VB') for _, tag in tagged)

#             if has_verb:
#                 events.append(sent)
        
#     return events

# ===========================VERSION 2===========================

import re

def classify_event(sentence):
    s = sentence.lower()

    # TRUE death (action-based)
    # Dialogue guard
    if '"' in sentence:
        return "dialogue", 1

    if (
        any(word in s for word in ["fell", "lay", "collapsed"]) and
        any(word in s for word in ["body", "blood", "snow", "ground"])
    ):
        return "death", 4

    # The Jon Snow Edge Case (Resurrection detection)
    if any(word in s for word in ["rose", "stirred", "stood up", "awoke"]) and \
       any(word in s for word in ["dead", "body", "corpse", "pale", "eyes"]):
        return "resurrection", 4

    # Talking ABOUT death ≠ death
    if "dead" in s and not any(word in s for word in ["fell", "killed", "slain"]):
        return "dialogue", 1
    
    # Atmosphere (lowest importance)
    if any(word in s for word in ["wind", "trees", "rustle", "cold", "snow"]) and \
        not any(word in s for word in ["strike", "slash", "attack", "hit"]):
        return "atmosphere", 1
    
    if any(word in s for word in ["sword", "blade", "iron"]) and \
        not any(word in s for word in ["raised", "swung", "slashed"]):
        return "description", 1

    if any(word in s for word in ["slashed", "struck", "hit", "checked", "fell back"]):
        return "combat", 3

    if any(word in s for word in ["sword", "blow", "blade", "strike"]):
        return "combat", 3

    if any(word in s for word in ["saw", "found", "noticed", "appeared"]):
        return "discovery", 3

    if '"' in sentence:
        return "dialogue", 2

    return "description", 1

def deduplicate_events(events):
    seen = set()
    unique = []

    for e in events:
        key = e["text"][:50]  # rough semantic fingerprint
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique

def extract_events(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    events = []

    for idx, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent) < 20:
            continue

        event_type, importance = classify_event(sent)

        events.append({
            "text": sent,
            "type": event_type,
            "importance": importance,
            "position": idx   # 🔥 NEW
        })

    events = deduplicate_events(events)
    return events

def get_event_threshold(days_gap):
    if days_gap > 90:
        return 4      # only critical events
    elif days_gap > 30:
        return 3
    elif days_gap > 7:
        return 2
    else:
        return 1      # everything
    
def get_event_limit(days_gap):
    if days_gap > 90:
        return 3
    elif days_gap > 30:
        return 5
    elif days_gap > 7:
        return 8
    else:
        return 15