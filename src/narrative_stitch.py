import re

# Legacy Stitching logic used by memory_render
def stitch_events(events, flags, mode="full"):
    # Rule: Death always stands alone
    for e in events:
        if e.get("type", "").upper() == "DEATH":
            return f"{e['text']}"

    texts = [e["text"] for e in events]

    if mode == "summary":
        return texts[0] if texts else ""

    return " ".join(texts)

# ---------------------------------------------------------
# ALGORITHM A: Template Transition Stitcher (Rule-based)
# ---------------------------------------------------------
class TemplateTransitionStitcher:
    """Uses a transition matrix based on event type shifts to form narratives."""
    
    def __init__(self):
        self.transitions = {
            ("EVENT", "COMBAT"): "Suddenly, tension flared causing a confrontation:",
            ("EVENT", "DISCOVERY"): "Following this, an important detail emerged:",
            ("DISCOVERY", "COMBAT"): "This realization quickly escalated into violence.",
            ("COMBAT", "DEATH"): "The brutal struggle culminated in a tragic loss.",
            ("COMBAT", "EVENT"): "After the clash, the dust settled.",
        }

    def stitch(self, memories):
        if not memories: return ""
        
        stitched = []
        last_tag = None
        
        for m in memories:
            # Handle if old code sends strings instead of dicts
            if isinstance(m, str):
                tag = re.search(r"\[(.*?)\]", m)
                tag = tag.group(1).upper() if tag else "EVENT"
                text = re.sub(r"\[.*?\]\s*", "", m).strip()
            else:
                tag = (m.get("tag") or "EVENT").upper()
                text = m.get("text", "")
                text = re.sub(r"\[.*?\]\s*", "", text).strip()
            
            if not text:
                continue
                
            if last_tag and last_tag != tag:
                transition = self.transitions.get((last_tag, tag), "")
                if transition:
                    stitched.append(transition)
            
            stitched.append(text)
            last_tag = tag
            
        return " ".join(stitched)

# ---------------------------------------------------------
# ALGORITHM B: Syntactic Fusion Stitcher (NLP Custom Heuristic)
# ---------------------------------------------------------
class SyntaxFusionStitcher:
    """Fuses sentences that share the same starting subject (naive pronoun/name merge)."""
    
    def stitch(self, memories):
        if not memories: return ""
        
        fused_sentences = []
        last_subject = None
        current_sentence = ""
        
        for m in memories:
            if isinstance(m, str):
                text = re.sub(r"\[.*?\]\s*", "", m).strip()
            else:
                text = m.get("text", "")
                text = re.sub(r"\[.*?\]\s*", "", text).strip()
                
            if not text: continue
            
            words = text.split()
            if not words: continue
            
            # Very naive subject extraction (first word, e.g. "He", "Jon", "The")
            subject = words[0]
            if len(words) > 1 and subject.lower() in ["the", "a", "an"]:
                subject = words[0] + " " + words[1]
                
            if current_sentence:
                if last_subject and subject.lower() == last_subject.lower() and len(words) > 1:
                    # Fuse
                    predicate = " ".join(words[1:]) if subject == words[0] else " ".join(words[2:])
                    # drop the capital letter of sequence
                    predicate = predicate[0].lower() + predicate[1:] if predicate else ""
                    # strip trailing punctuation from previous
                    if current_sentence[-1] in [".", "!", "?"]:
                        current_sentence = current_sentence[:-1]
                    current_sentence += f" and {predicate}"
                    if not current_sentence.endswith("."): current_sentence += "."
                else:
                    fused_sentences.append(current_sentence)
                    current_sentence = text
                    last_subject = subject
            else:
                current_sentence = text
                last_subject = subject
                
        if current_sentence:
            fused_sentences.append(current_sentence)
            
        return " ".join(fused_sentences)

# ---------------------------------------------------------
# ALGORITHM C: Lexical Pacing Stitcher (Metadata Driven)
# ---------------------------------------------------------
class LexicalPacingStitcher:
    """Uses chronological index gaps to insert temporal pacing words."""
    
    def stitch(self, memories):
        if not memories: return ""
        
        stitched = []
        last_index = None
        
        for m in memories:
            if isinstance(m, str):
                text = re.sub(r"\[.*?\]\s*", "", m).strip()
                idx = 0
                tag = "EVENT"
            else:
                text = m.get("text", "")
                text = re.sub(r"\[.*?\]\s*", "", text).strip()
                idx = m.get("index", 0)
                tag = (m.get("tag") or "EVENT").upper()
            
            if not text: continue
            
            if last_index is not None and idx is not None and last_index is not None:
                gap = idx - last_index
                if gap > 20:
                    stitched.append("Much later,")
                elif gap > 5:
                    stitched.append("Sometime after,")
                elif gap <= 1 and tag == "COMBAT":
                    stitched.append("Immediately,")
            
            stitched.append(text)
            last_index = idx
            
        return " ".join(stitched)

def test_stitchers(memories):
    """Utility to run all 3 stitchers and return their outputs."""
    a = TemplateTransitionStitcher().stitch(memories)
    b = SyntaxFusionStitcher().stitch(memories)
    c = LexicalPacingStitcher().stitch(memories)
    return {"A_Template": a, "B_Syntax": b, "C_Pacing": c}

