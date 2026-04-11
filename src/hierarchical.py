from src.summarizer import summarize_text

def summarize_paragraphs(paragraphs):
    """
    Summarize each paragraph into 1 sentence.
    """
    paragraph_summaries = []

    for para in paragraphs:
        summary = summarize_text(para, summary_level="very_short")
        paragraph_summaries.append(summary)

    return paragraph_summaries

def summarize_hierarchy(paragraph_summaries, level):
    """
    Summarize the summaries (Level-2 hierarchy).
    """
    combined_text = " ".join(paragraph_summaries)

    # Map hierarchy depth
    if level == "very_short":
        return summarize_text(combined_text, "very_short")
    elif level == "short":
        return summarize_text(combined_text, "short")
    elif level == "medium":
        return summarize_text(combined_text, "medium")
    else:  # long
        return summarize_text(combined_text, "long")