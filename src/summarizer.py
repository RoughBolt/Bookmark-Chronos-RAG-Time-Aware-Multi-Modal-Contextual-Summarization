import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

def summarize_text(text, summary_level):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]

    word_freq = Counter(words)

    sentence_scores = {}

    for sent in sentences:
        sent_words = word_tokenize(sent.lower())
        score = sum(word_freq.get(w, 0) for w in sent_words)
        sentence_scores[sent] = score

    if summary_level == 'very_short':
        ratio = 0.02
    elif summary_level == "short":
        ratio = 0.05
    elif summary_level == "medium":
        ratio = 0.1
    else:
        ratio = 0.2

    summary_len = max(1, int(len(sentences) * ratio))

    top_sentences = sorted(
        sentence_scores,
        key = sentence_scores.get,
        reverse = True
    )[:summary_len]

    top_sentences.sort(key=lambda s: sentences.index(s))

    return " ".join(top_sentences)