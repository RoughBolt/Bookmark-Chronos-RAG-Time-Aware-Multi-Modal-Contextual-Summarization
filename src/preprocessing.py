import re

# from utils import (                         # For Driver
#     find_chapters,                          # For Driver 
#     extract_text_upto_chapter,              # For Driver
# )
# from bookmark import load_bookmark          # For Driver

def clean_text(text):
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Remove metadata & noise
        if (
            "A Game Of Thrones" in line or
            "George R. R. Martin" in line or
            "Book One of A Song of Ice and Fire" in line or
            line.startswith("Page ") or
            line.isupper()
        ):
            continue

        cleaned.append(line)

    text = " ".join(cleaned)
    text = re.sub(r'\s+', ' ', text)

    return text


# BOOK_PATH = "data/book.txt"                                 #Driver
# chapters = find_chapters(BOOK_PATH)                         #Driver
# bookmark = load_bookmark()                                  #Driver
# text = extract_text_upto_chapter(                           #Driver
#     BOOK_PATH,                                              #Driver
#     chapters,                                               #Driver    
#     bookmark["pov"],                                        #Driver
#     bookmark["occurrence"]                                  #Driver
# )                                                           #Driver
# clean_text(text)                                            #Driver 