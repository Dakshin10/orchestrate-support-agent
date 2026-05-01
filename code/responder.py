import re
from typing import List, Tuple

MIN_WORDS = 6
MIN_OVERLAP = 1


def _tokens(text: str) -> set:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _clean_doc(text: str) -> str:
    text = re.sub(r"---.*?---", "", text, flags=re.DOTALL)
    text = re.sub(r"(title|slug|url|last_updated).*?:.*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#*_`>|~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?]) +", text)


def _is_quality_sentence(s: str, query_tokens: set) -> bool:
    words = s.split()
    s_lower = s.lower()

    if len(words) < MIN_WORDS:
        return False

    if s.endswith("?"):
        return False

    if any(x in s_lower for x in [
        "last updated", "faq", "overview", "example",
        "leaderboard", "template", "related articles"
    ]):
        return False

    if "-" in s or ":" in s:
        return False

    if re.match(r"^(click|follow|select|choose|go to|navigate|open)\b", s_lower):
        return False

    if any(x in s_lower for x in [
        "will help you", "this article", "important aspects"
    ]):
        return False

    if any(x in s_lower for x in [
        "travel with visa", "enjoy visa", "adventures"
    ]):
        return False

    if "anthropic uses different robots" in s_lower:
        return False

    # reject fragments (but allow "you can...")
    if s[0].islower() and not s.startswith("you "):
        return False

    overlap = len(query_tokens & _tokens(s))
    return overlap >= MIN_OVERLAP


class Responder:

    def generate(self, query: str, docs: List[str],
                 max_chars: int = 180) -> Tuple[str, float]:

        if not docs:
            return "", 0.0

        query_tokens = _tokens(query)

        best_sentence = ""
        best_score = 0

        for doc in docs:
            clean = _clean_doc(doc)
            sentences = _split_sentences(clean)

            for s in sentences:
                s = s.strip()

                if not _is_quality_sentence(s, query_tokens):
                    continue

                score = len(query_tokens & _tokens(s))

                if score > best_score:
                    best_score = score
                    best_sentence = s

        # fallback — ensure FULL sentence
        if not best_sentence:
            for doc in docs:
                clean = _clean_doc(doc)
                sentences = _split_sentences(clean)

                for s in sentences:
                    s = s.strip()
                    if len(s.split()) > 8 and s.endswith("."):
                        return s, 0.3

            return "", 0.0

        # remove headings
        best_sentence = re.sub(
            r"^[A-Z][A-Za-z ]{3,25}\s+(?=You|If|When)",
            "",
            best_sentence
        )

        # simplify Visa verbosity
        best_sentence = re.sub(
            r"\(only free when dialling from within the USA\)",
            "",
            best_sentence
        )

        # capitalize
        best_sentence = best_sentence[0].upper() + best_sentence[1:]

        # ensure period
        if not best_sentence.endswith("."):
            best_sentence += "."

        # 🔥 IMPORTANT: DO NOT TRUNCATE — preserve full sentence
        if len(best_sentence) > max_chars:
            sentences = re.split(r'(?<=[.!?]) +', best_sentence)
            for s in sentences:
                if len(s) <= max_chars:
                    best_sentence = s
                    break

        return best_sentence, round(best_score / 5, 3)


_responder = None


def generate_response_with_confidence(query: str, docs: List[str]):
    global _responder
    if _responder is None:
        _responder = Responder()
    return _responder.generate(query, docs)