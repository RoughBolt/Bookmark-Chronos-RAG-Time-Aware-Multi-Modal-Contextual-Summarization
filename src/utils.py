from datetime import datetime

# What this function does
# Read book.txt
# Find lines that are:
# ALL CAPS
# Short (1–2 words)
# Treat them as chapter headings
# Store their position

def find_chapters(book_path):
    chapters = []
    counts = {}

    with open (book_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        clean = line.strip()
    
        if clean.isupper() and 0 < len(clean.split()) <=2:
            counts[clean] = counts.get(clean, 0) + 1
            chapters.append((clean, counts[clean], i))
    
    return chapters

# What this function does:
# Takes chapters list
# Takes a target (POV, occurrence)
# Returns the text slice

# start = 0
# end = line_index_of_target_chapter
# text = lines[start:end]

def extract_text_upto_chapter(book_path, chapters, target_pov, target_occurrence):
    end_line = None

    for name, occ, line_idex in chapters:
        if name ==target_pov and occ == target_occurrence:
            end_line = line_idex
            break

    if end_line is None:
        return ValueError("Target chapter not found")
    
    with open(book_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    extracted_text = "".join(lines[0:end_line])

    return extracted_text

# What these functions do:
# Compute days since last read
# Convert that into SUMMARY DEPTH LEVELS

def days_since_last_read(last_read_str):
    """
    last_read_str format: YYYY-MM-DD
    """
    last_read_date = datetime.strptime(last_read_str, "%Y-%m-%d")
    today = datetime.today()

    delta = today - last_read_date
    return delta.days

def get_summary_level(days_gap):
    """
    Returns a summary depth level based on time gap
    """
    if days_gap <= 3:
        return "very_short"
    elif days_gap <= 14:
        return "short"
    elif days_gap <= 90:
        return "medium"
    else:
        return "long"
    