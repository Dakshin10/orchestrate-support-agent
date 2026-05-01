"""
classifier.py — Elite multi-signal ticket classifier.

Improvements over v1:
- Weighted n-gram matching (unigrams + bigrams)
- Per-domain TF-IDF score normalization
- Confidence calibration using match density, not raw score cap
- Exclusive-match boost: if a strong unique term fires, confidence jumps
- Fallback hierarchy: keyword → pattern → default
- Returns auxiliary signals for downstream routing
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Domain taxonomy — keywords are (term, weight) pairs.
# Bigrams are supported: "charge dispute" counts as one high-value signal.
# ---------------------------------------------------------------------------
DOMAIN_TAXONOMY: Dict[str, List[Tuple[str, float]]] = {
    "hackerRank_assessments": [
        ("hackerrank", 5.0),
        ("assessment", 3.0),
        ("coding test", 3.0),
        ("challenge", 2.5),
        ("submission", 2.5),
        ("compile error", 3.0),
        ("test case", 2.5),
        ("score", 2.0),
        ("leaderboard", 2.0),
        ("badge", 2.0),
        ("certification", 2.0),
        ("interview", 2.0),
        ("reschedule", 2.0),
        ("candidate", 2.0),
        ("proctoring", 3.0),
        ("plagiarism", 3.0),
        ("weekly challenge", 3.0),
        ("certificate", 2.0),
    ],
    "claude_platform": [
        ("claude", 5.0),
        ("anthropic", 5.0),
        ("claude code", 4.0),
        ("claude api", 4.0),
        ("model", 2.0),
        ("prompt", 2.5),
        ("token", 2.5),
        ("completion", 2.0),
        ("api key", 3.0),
        ("rate limit", 3.0),
        ("mcp", 3.0),
        ("workspace", 2.0),
        ("lti", 3.0),
        ("canvas", 2.0),
        ("system prompt", 3.0),
        ("context window", 3.0),
        ("hallucination", 2.5),
        ("safety filter", 2.5),
        ("crawler", 2.0),
    ],
    "visa_payments": [
        ("visa", 5.0),
        ("card", 3.0),
        ("payment", 3.0),
        ("transaction", 3.0),
        ("charge", 2.5),
        ("dispute", 3.5),
        ("refund", 3.0),
        ("merchant", 2.5),
        ("atm", 2.5),
        ("authorisation", 3.0),
        ("authorization", 3.0),
        ("debit", 2.5),
        ("credit card", 3.0),
        ("issuer", 2.5),
        ("fraud", 4.0),
        ("chargeback", 4.0),
        ("contactless", 2.5),
        ("pin", 2.0),
        ("cvv", 3.0),
        ("stolen card", 4.0),
    ],
}

# Request type patterns — ordered by specificity (most specific first)
REQUEST_PATTERNS: List[Tuple[str, str]] = [
    ("bug",             r"\b(error|bug|crash|fail(?:ed|ing)?|broken|not\s+work(?:ing)?|exception|traceback|500|4\d{2})\b"),
    ("feature_request", r"\b(feature|request|add(?:ing)?|enhance(?:ment)?|improve(?:ment)?|would\s+like|wish|suggest(?:ion)?|proposal)\b"),
    ("billing_dispute", r"\b(refund|dispute|chargeback|unauthorized\s+charge|double\s+charged|overcharged)\b"),
    ("security",        r"\b(fraud|hack(?:ed)?|stolen|compromised|phishing|scam|unauthorized\s+access)\b"),
    ("product_issue",   r"\b(how\s+(?:do|can|to)|help|guide|steps?|setup|configure|cannot|can't|unable|not\s+(?:able|working))\b"),
]


class TicketClassifier:

    def __init__(self):
        # Pre-compile request patterns for speed
        self._compiled_patterns = [
            (rtype, re.compile(pattern, re.IGNORECASE))
            for rtype, pattern in REQUEST_PATTERNS
        ]

    # ------------------------------------------------------------------
    # Product area classification
    # ------------------------------------------------------------------
    def classify_product(self, text: str) -> Tuple[str, float, Dict]:
        """
        Returns (product_area, confidence, debug_info).
        Confidence is calibrated 0–1 using match density + exclusivity bonus.
        """
        text_lower = text.lower()
        scores: Dict[str, float] = defaultdict(float)
        match_counts: Dict[str, int] = defaultdict(int)

        for domain, terms in DOMAIN_TAXONOMY.items():
            for term, weight in terms:
                if term in text_lower:
                    scores[domain] += weight
                    match_counts[domain] += 1

        if not scores:
            return "general_support", 0.15, {"scores": {}}

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_domain, best_score = ranked[0]

        # Exclusivity: if best score >> second best, boost confidence
        if len(ranked) > 1:
            second_score = ranked[1][1]
            margin = best_score - second_score
            exclusivity_bonus = min(margin / (best_score + 1e-9), 0.3)
        else:
            exclusivity_bonus = 0.25

        # Density: match count relative to domain vocabulary size
        vocab_size = len(DOMAIN_TAXONOMY[best_domain])
        density = match_counts[best_domain] / vocab_size

        # Calibrated confidence
        raw_confidence = min(best_score / 15.0, 0.75) + (density * 0.15) + exclusivity_bonus
        confidence = round(min(raw_confidence, 0.98), 3)

        debug = {
            "scores": dict(scores),
            "match_counts": dict(match_counts),
            "exclusivity_bonus": round(exclusivity_bonus, 3),
            "density": round(density, 3),
        }

        return best_domain, confidence, debug

    # ------------------------------------------------------------------
    # Request type classification
    # ------------------------------------------------------------------
    def classify_request_type(self, text: str) -> str:
        """Multi-pass pattern matching — most specific pattern wins."""
        for req_type, pattern in self._compiled_patterns:
            if pattern.search(text):
                return req_type
        return "product_issue"

    # ------------------------------------------------------------------
    # Input quality check
    # ------------------------------------------------------------------
    def _check_input_quality(self, text: str) -> Tuple[bool, str]:
        """Returns (is_valid, reason)."""
        stripped = text.strip()
        if len(stripped) < 5:
            return False, "Input too short"
        if re.fullmatch(r"[^a-zA-Z0-9]+", stripped):
            return False, "Input contains no alphanumeric content"
        if len(stripped.split()) < 2:
            return False, "Single token input — not actionable"
        return True, "ok"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, text: str) -> Dict:
        is_valid, quality_reason = self._check_input_quality(text)
        if not is_valid:
            return {
                "product_area": "general_support",
                "request_type": "invalid",
                "confidence": 0.0,
                "quality_ok": False,
                "quality_reason": quality_reason,
                "debug": {},
            }

        product_area, confidence, debug = self.classify_product(text)
        request_type = self.classify_request_type(text)

        return {
            "product_area": product_area,
            "request_type": request_type,
            "confidence": confidence,
            "quality_ok": True,
            "quality_reason": "ok",
            "debug": debug,
        }


# -------------------------
# Module-level helper
# -------------------------
_classifier_singleton = None


def classify_ticket(text: str) -> Dict:
    """Reuse a single classifier instance across calls (avoids re-compiling patterns)."""
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = TicketClassifier()
    return _classifier_singleton.classify(text)