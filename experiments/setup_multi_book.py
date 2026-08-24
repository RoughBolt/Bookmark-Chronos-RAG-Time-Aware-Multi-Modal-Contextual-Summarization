"""
Multi-Document Generalization Setup
=====================================
Downloads and preprocesses public-domain books from Project Gutenberg
to test Chronos-RAG beyond a single novel.

Validates that the pipeline generalizes across:
  1. Different genres (fantasy, mystery, adventure, literary fiction)
  2. Different writing styles (19th-century vs modern)
  3. Different narrative structures (first-person, third-person, multi-POV)

Books selected for complementary properties:
  - Frankenstein:     Gothic horror, first-person epistolary, death/resurrection themes
  - Sherlock Holmes:  Mystery, third-person, temporal reasoning critical
  - Pride & Prejudice: Literary fiction, dialogue-heavy, subtle character development
  - Dracula:          Horror, epistolary multi-POV, temporal diary entries

Each book gets its own bookmark.json, QA pairs, and evaluation set.

Usage:
  python -m experiments.setup_multi_book
  python -m experiments.setup_multi_book --book frankenstein
  python -m experiments.setup_multi_book --list

Requirements:
  Internet access (downloads from Project Gutenberg)
"""

import argparse
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Book catalog ──────────────────────────────────────────────────────────────

BOOKS = {
    "frankenstein": {
        "title": "Frankenstein; or, The Modern Prometheus",
        "author": "Mary Shelley",
        "year": 1818,
        "genre": "gothic_horror",
        "gutenberg_id": 84,
        "narrative_style": "first_person_epistolary",
        "why": "Death/resurrection themes directly test event classifier. "
               "Epistolary format tests chapter extraction adaptability.",
        "chapter_pattern": r"^(?:Chapter|CHAPTER|Letter)\s+\w+",
        "qa_pairs": [
            {
                "id": "F01", "chapter": "LETTER_1", "question_type": "factual",
                "question": "Who is writing the opening letters and to whom?",
                "ground_truth": "Robert Walton is writing letters to his sister Margaret Saville. He is an explorer on a voyage to the North Pole."
            },
            {
                "id": "F02", "chapter": "CHAPTER_1", "question_type": "factual",
                "question": "Where did Victor Frankenstein grow up?",
                "ground_truth": "Victor grew up in Geneva, Switzerland, in a wealthy and loving family. His father was Alphonse Frankenstein."
            },
            {
                "id": "F03", "chapter": "CHAPTER_5", "question_type": "factual",
                "question": "How did Victor react when his creation first came to life?",
                "ground_truth": "Victor was horrified by his creation. Despite months of work, when the creature opened its dull yellow eyes, Victor was disgusted and fled in terror, abandoning the creature."
            },
            {
                "id": "F04", "chapter": "CHAPTER_7", "question_type": "causal",
                "question": "Who was William Frankenstein and what happened to him?",
                "ground_truth": "William was Victor's youngest brother. He was murdered by the creature as an act of revenge against Victor. The creature strangled the boy."
            },
            {
                "id": "F05", "chapter": "CHAPTER_8", "question_type": "multi_hop",
                "question": "Why was Justine accused of William's murder?",
                "ground_truth": "Justine Moritz was falsely accused because the creature planted William's locket — a miniature portrait — on her. She was convicted and executed despite her innocence, adding to Victor's guilt."
            },
            {
                "id": "F06", "chapter": "CHAPTER_10", "question_type": "character_state",
                "question": "What did the creature demand from Victor?",
                "ground_truth": "The creature demanded that Victor create a female companion for him, arguing that his loneliness and rejection by society drove him to violence. He promised to disappear with his mate if Victor complied."
            },
            {
                "id": "F07", "chapter": "CHAPTER_20", "question_type": "causal",
                "question": "Why did Victor destroy the female creature he was making?",
                "ground_truth": "Victor destroyed the female creature because he feared creating a race of monsters. He worried they might breed and terrorize humanity, or that the female might reject the male creature."
            },
            {
                "id": "F08", "chapter": "CHAPTER_21", "question_type": "multi_hop",
                "question": "What was the connection between Victor's refusal and Henry Clerval's death?",
                "ground_truth": "When Victor destroyed the female creature, the monster swore revenge and killed Victor's best friend Henry Clerval on the same night. This was the creature's retaliation for Victor breaking his promise."
            },
            {
                "id": "F09", "chapter": "CHAPTER_23", "question_type": "temporal",
                "question": "What happened on Victor's wedding night?",
                "ground_truth": "The creature killed Elizabeth, Victor's bride, on their wedding night. Despite Victor's attempts to protect her, the creature strangled Elizabeth, fulfilling his threat to be with Victor on his wedding night."
            },
            {
                "id": "F10", "chapter": "CHAPTER_24", "question_type": "character_state",
                "question": "How does the story of Victor Frankenstein end?",
                "ground_truth": "Victor died aboard Walton's ship while pursuing the creature through the Arctic. After Victor's death, the creature appeared and expressed remorse, then declared he would end his own life on a funeral pyre."
            },
        ],
    },
    "sherlock_holmes": {
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "year": 1892,
        "genre": "mystery",
        "gutenberg_id": 1661,
        "narrative_style": "first_person_watson",
        "why": "Temporal reasoning is critical for mystery plots. "
               "Tests whether decay model correctly preserves clues.",
        "chapter_pattern": r"^(?:ADVENTURE|CHAPTER|I+V?X?\.)",
        "qa_pairs": [
            {
                "id": "SH01", "chapter": "ADVENTURE_1", "question_type": "factual",
                "question": "What was the scandal involving Irene Adler?",
                "ground_truth": "In 'A Scandal in Bohemia,' the King of Bohemia hired Holmes to recover a compromising photograph of himself with Irene Adler, an opera singer. Adler outwitted Holmes and kept the photo, earning his lasting admiration."
            },
            {
                "id": "SH02", "chapter": "ADVENTURE_1", "question_type": "character_state",
                "question": "How did Sherlock Holmes regard Irene Adler?",
                "ground_truth": "Holmes referred to Adler as 'the woman,' the only woman who ever outsmarted him. She was the one person who earned his genuine respect and admiration, though his feelings were not romantic."
            },
            {
                "id": "SH03", "chapter": "ADVENTURE_2", "question_type": "causal",
                "question": "What was the mystery of the Red-Headed League?",
                "ground_truth": "The Red-Headed League was a fake organization created by criminal John Clay to lure pawnbroker Jabez Wilson out of his shop. While Wilson was away copying the encyclopedia, Clay dug a tunnel from Wilson's cellar to the bank next door."
            },
            {
                "id": "SH04", "chapter": "ADVENTURE_3", "question_type": "multi_hop",
                "question": "How did Holmes solve the case of identity in 'A Case of Identity'?",
                "ground_truth": "Holmes deduced that the mysterious Hosmer Angel was actually Mary Sutherland's stepfather in disguise. The stepfather created the fake suitor to keep Mary single and continue collecting her inheritance income."
            },
            {
                "id": "SH05", "chapter": "ADVENTURE_5", "question_type": "factual",
                "question": "What was unique about the Five Orange Pips case?",
                "ground_truth": "The case involved a series of murders connected to the Ku Klux Klan, with victims receiving envelopes containing five orange pips as a death warning. It was one of the few cases Holmes considered a failure."
            },
        ],
    },
    "pride_prejudice": {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "year": 1813,
        "genre": "literary_fiction",
        "gutenberg_id": 1342,
        "narrative_style": "third_person_omniscient",
        "why": "Dialogue-heavy text challenges event classifier on dialogue vs. "
               "character development. Subtle relationship arcs test temporal decay.",
        "chapter_pattern": r"^Chapter\s+\d+",
        "qa_pairs": [
            {
                "id": "PP01", "chapter": "CHAPTER_1", "question_type": "factual",
                "question": "What news excites Mrs. Bennet at the beginning of Pride and Prejudice?",
                "ground_truth": "Mrs. Bennet is excited that Netherfield Park has been let at last to Mr. Bingley, a single man of large fortune, whom she immediately considers a prospective husband for one of her five daughters."
            },
            {
                "id": "PP02", "chapter": "CHAPTER_3", "question_type": "character_state",
                "question": "What was Elizabeth Bennet's first impression of Mr. Darcy?",
                "ground_truth": "Elizabeth found Darcy proud and disagreeable. At the Meryton ball, Darcy refused to dance and was overheard saying Elizabeth was 'tolerable but not handsome enough to tempt me,' deeply offending her."
            },
            {
                "id": "PP03", "chapter": "CHAPTER_18", "question_type": "causal",
                "question": "Why did Mr. Darcy separate Bingley from Jane?",
                "ground_truth": "Darcy believed Jane did not truly care for Bingley and that the Bennet family's lack of propriety made them unsuitable. He convinced Bingley to leave Netherfield, causing Jane great heartbreak."
            },
            {
                "id": "PP04", "chapter": "CHAPTER_34", "question_type": "multi_hop",
                "question": "What happened during Darcy's first proposal and why was it rejected?",
                "ground_truth": "Darcy proposed to Elizabeth at Hunsford but insulted her family throughout. Elizabeth rejected him furiously, accusing him of separating Jane and Bingley, and of ruining Wickham. Darcy was shocked by the refusal."
            },
            {
                "id": "PP05", "chapter": "CHAPTER_46", "question_type": "multi_hop",
                "question": "How did the Lydia-Wickham crisis ultimately bring Elizabeth and Darcy together?",
                "ground_truth": "When Lydia eloped with Wickham, Darcy secretly paid Wickham's debts and arranged the marriage to save the Bennet family's honor. When Elizabeth learned of his intervention, her feelings toward him fundamentally changed, leading to their eventual engagement."
            },
        ],
    },
    "dracula": {
        "title": "Dracula",
        "author": "Bram Stoker",
        "year": 1897,
        "genre": "gothic_horror",
        "gutenberg_id": 345,
        "narrative_style": "epistolary_multi_pov",
        "why": "Epistolary diary format with dates tests temporal awareness. "
               "Death/undeath themes are ideal for event classifier evaluation.",
        "chapter_pattern": r"^(?:CHAPTER|Chapter)\s+\w+",
        "qa_pairs": [
            {
                "id": "DR01", "chapter": "CHAPTER_1", "question_type": "factual",
                "question": "Where did Jonathan Harker travel to and why?",
                "ground_truth": "Jonathan Harker traveled to Transylvania to finalize a real estate transaction with Count Dracula, who was purchasing property in England. He stayed at Castle Dracula in the Carpathian Mountains."
            },
            {
                "id": "DR02", "chapter": "CHAPTER_3", "question_type": "character_state",
                "question": "What did Harker discover about Count Dracula during his stay?",
                "ground_truth": "Harker discovered that Dracula cast no reflection in mirrors, slept during the day, could climb walls like a lizard, and was effectively imprisoning Harker in the castle. He realized Dracula was not human."
            },
            {
                "id": "DR03", "chapter": "CHAPTER_8", "question_type": "temporal",
                "question": "What happened to Lucy Westenra?",
                "ground_truth": "Lucy was bitten by Dracula and slowly drained of blood over several weeks. Despite blood transfusions from multiple donors and Van Helsing's efforts with garlic, Lucy died and rose as a vampire."
            },
            {
                "id": "DR04", "chapter": "CHAPTER_16", "question_type": "causal",
                "question": "How was vampire Lucy finally destroyed?",
                "ground_truth": "Lucy's fiancé Arthur Holmwood drove a stake through her heart while Van Helsing read prayers. Her head was cut off and her mouth filled with garlic. This freed her soul from the curse of undeath."
            },
            {
                "id": "DR05", "chapter": "CHAPTER_27", "question_type": "multi_hop",
                "question": "How was Count Dracula ultimately defeated?",
                "ground_truth": "The group tracked Dracula back to Transylvania. Jonathan Harker slashed Dracula's throat with a kukri knife while Quincey Morris stabbed him through the heart with a bowie knife. Dracula crumbled to dust. Quincey died from wounds received in the fight."
            },
        ],
    },
}


# ── Download and preprocessing ────────────────────────────────────────────────

def download_book(book_key: str, output_dir: str = "data/books") -> str:
    """Downloads a book from Project Gutenberg and saves it locally."""
    book = BOOKS[book_key]
    gid = book["gutenberg_id"]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{book_key}.txt")

    if os.path.exists(output_path):
        print(f"[MultiBook] {book['title']} already downloaded → {output_path}")
        return output_path

    # Project Gutenberg mirror URLs
    urls = [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    ]

    for url in urls:
        try:
            print(f"[MultiBook] Downloading {book['title']} from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "ChronosRAG/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")

            # Strip Project Gutenberg header/footer
            text = _strip_gutenberg_boilerplate(text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"[MultiBook] Saved → {output_path} ({len(text):,} chars)")
            return output_path

        except Exception as e:
            print(f"[MultiBook] Failed from {url}: {e}")
            continue

    raise RuntimeError(f"Could not download {book['title']} from any mirror")


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Removes Project Gutenberg headers and footers."""
    # Find start marker
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find end of the line
            newline = text.find("\n", idx)
            start_idx = newline + 1 if newline != -1 else idx + len(marker)
            break

    # Find end marker
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def create_book_config(book_key: str, output_dir: str = "data/books") -> dict:
    """Creates bookmark.json and qa_pairs.json for a downloaded book."""
    book = BOOKS[book_key]
    book_dir = os.path.join(output_dir, book_key)
    os.makedirs(book_dir, exist_ok=True)

    # Create bookmark (default to start of book)
    bookmark = {
        "book": book["title"],
        "pov": "ALL",
        "occurrence": 1,
        "last_read": "2026-07-01",
        "genre": book["genre"],
        "narrative_style": book["narrative_style"],
    }
    bookmark_path = os.path.join(book_dir, "bookmark.json")
    with open(bookmark_path, "w") as f:
        json.dump(bookmark, f, indent=2)

    # Create QA pairs
    qa_path = os.path.join(book_dir, "qa_pairs.json")
    with open(qa_path, "w") as f:
        json.dump(book["qa_pairs"], f, indent=2)

    print(f"[MultiBook] Config created: {bookmark_path}, {qa_path}")
    return {"bookmark": bookmark_path, "qa": qa_path}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and set up public-domain books for multi-document evaluation"
    )
    parser.add_argument("--book", choices=list(BOOKS.keys()),
                        help="Download a specific book (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="List available books and exit")
    parser.add_argument("--output-dir", default="data/books",
                        help="Output directory for book files")
    args = parser.parse_args()

    if args.list:
        print("\n  Available books for multi-document evaluation:\n")
        for key, book in BOOKS.items():
            print(f"  {key:>20}  |  {book['title']}")
            print(f"  {'':>20}  |  {book['author']} ({book['year']})")
            print(f"  {'':>20}  |  Genre: {book['genre']}, Style: {book['narrative_style']}")
            print(f"  {'':>20}  |  Why: {book['why']}")
            print(f"  {'':>20}  |  QA pairs: {len(book['qa_pairs'])}")
            print()
        return

    books_to_process = [args.book] if args.book else list(BOOKS.keys())

    print(f"\n[MultiBook] Setting up {len(books_to_process)} book(s)...\n")

    for book_key in books_to_process:
        try:
            path = download_book(book_key, args.output_dir)
            config = create_book_config(book_key, args.output_dir)
            print(f"  ✓ {BOOKS[book_key]['title']}")
        except Exception as e:
            print(f"  ✗ {BOOKS[book_key]['title']}: {e}")

    print(f"\n[MultiBook] Setup complete. Run ablation_study.py with --multi-book to evaluate.\n")


if __name__ == "__main__":
    main()
