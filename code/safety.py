"""
safety.py — Elite layered safety engine.

Improvements over v1:
- Three-tier risk matrix (critical / high / medium / low)
- Confidence-adjusted thresholds (low confidence widens escalation window)
- Context-aware: billing disputes in visa domain treated differently
- Composite risk score with diminishing returns on same-tier matches
- Full audit trail returned for traceability
- Configurable thresholds via RiskConfig
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Risk patterns — each entry: (label, compiled_regex, tier, score_contribution)
# Tiers: "critical" > "high" > "medium" > "low"
# ---------------------------------------------------------------------------
@dataclass
class RiskPattern:
    label: str
    pattern: re.Pattern
    tier: str
    score: float


def _p(label: str, raw: str, tier: str, score: float) -> RiskPattern:
    return RiskPattern(label, re.compile(raw, re.IGNORECASE), tier, score)


RISK_PATTERNS: List[RiskPattern] = [
    # --- Critical (auto-escalate regardless of confidence) ---
    _p("account_compromised",       r"\baccount\s+(compromised|hacked|taken\s+over)\b",  "critical", 10.0),
    _p("unauthorized_transaction",  r"\bunauthorized\s+(transaction|charge|access)\b",   "critical", 10.0),
    _p("identity_theft",            r"\bidentity\s+theft\b",                              "critical", 10.0),
    _p("data_breach",               r"\bdata\s+breach\b",                                 "critical", 10.0),

    # --- High ---
    _p("fraud",                     r"\bfraud(?:ulent)?\b",                               "high", 7.0),
    _p("stolen_card",               r"\b(stolen|lost)\s+(card|wallet)\b",                 "high", 7.0),
    _p("scam",                      r"\bscam\b",                                           "high", 6.0),
    _p("hacked",                    r"\bhack(?:ed|ing)?\b",                               "high", 6.5),
    _p("phishing",                  r"\bphish(?:ing|ed)?\b",                              "high", 7.0),
    _p("threat",                    r"\b(threat|threaten|sue|legal\s+action)\b",          "high", 5.5),

    # --- Medium ---
    _p("refund",                    r"\brefund\b",                                         "medium", 3.0),
    _p("billing_issue",             r"\bbilling\s+(issue|problem|error)\b",               "medium", 3.5),
    _p("payment_failed",            r"\bpayment\s+fail(?:ed|ure)?\b",                    "medium", 3.0),
    _p("account_locked",            r"\baccount\s+(locked|suspended|disabled|blocked)\b", "medium", 4.0),
    _p("cannot_login",              r"\b(cannot|can't|unable\s+to)\s+(log\s*in|sign\s*in|access)\b", "medium", 3.0),
    _p("charge_dispute",            r"\bcharge\s+dispute\b",                              "medium", 4.0),
    _p("double_charge",             r"\b(double|duplicate)\s+charg(e|ed)\b",             "medium", 4.0),

    # --- Low (safe signals — reduce score) ---
    _p("how_to",                    r"\bhow\s+(do|can|to)\b",                             "low", -1.0),
    _p("documentation",             r"\b(guide|docs|documentation|tutorial|steps)\b",    "low", -0.5),
    _p("general_question",          r"\bwhat\s+is\b|\bcan\s+you\s+help\b",               "low", -0.5),
]


@dataclass
class RiskConfig:
    """Tunable thresholds."""
    critical_auto_escalate: bool = True
    high_auto_escalate: bool = True
    medium_low_confidence_threshold: float = 0.5   # escalate medium if confidence < this
    medium_score_threshold: float = 4.0             # escalate if composite score >= this
    composite_score_threshold: float = 8.0          # always escalate above this regardless


class SafetyEngine:

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    # ------------------------------------------------------------------
    # Scan text against all patterns
    # ------------------------------------------------------------------
    def _scan(self, text: str) -> Tuple[List[Dict], float]:
        """Returns (matched_patterns, composite_score)."""
        matched = []
        raw_score = 0.0
        tier_counts: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for rp in RISK_PATTERNS:
            if rp.pattern.search(text):
                # Diminishing returns for same-tier matches beyond the first
                same_tier = tier_counts[rp.tier]
                diminishing = 1.0 / (1 + same_tier * 0.4)
                contribution = rp.score * diminishing

                matched.append({
                    "label": rp.label,
                    "tier": rp.tier,
                    "contribution": round(contribution, 2),
                })
                raw_score += contribution
                tier_counts[rp.tier] += 1

        composite = round(raw_score, 2)
        return matched, composite

    # ------------------------------------------------------------------
    # Determine highest tier matched
    # ------------------------------------------------------------------
    def _highest_tier(self, matched: List[Dict]) -> str:
        tier_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        if not matched:
            return "none"
        return max(matched, key=lambda m: tier_order.get(m["tier"], 0))["tier"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def should_escalate(self, text: str, confidence: float = 1.0,
                        product_area: str = "general_support") -> Dict:
        """
        Returns escalation decision with full audit trail.
        """
        text_lower = text.lower()
        matched, composite = self._scan(text_lower)
        highest = self._highest_tier(matched)
        cfg = self.config

        # --- Decision logic ---
        escalate = False
        reason = "No significant risk detected."

        if highest == "critical" and cfg.critical_auto_escalate:
            escalate = True
            reason = "Critical-tier risk detected — auto-escalated (potential account compromise/fraud)."

        elif highest == "high" and cfg.high_auto_escalate:
            escalate = True
            reason = "High-risk issue detected (security/fraud related)."

        elif highest == "medium":
            if confidence < cfg.medium_low_confidence_threshold:
                escalate = True
                reason = f"Medium-risk issue with low classification confidence ({confidence:.2f})."
            elif composite >= cfg.medium_score_threshold:
                escalate = True
                reason = f"Medium-risk composite score ({composite}) exceeds threshold ({cfg.medium_score_threshold})."

        elif composite >= cfg.composite_score_threshold:
            escalate = True
            reason = f"Elevated composite risk score ({composite})."

        # product_area context modifier: financial disputes in non-visa domain = suspicious
        if (not escalate and "dispute" in text_lower and product_area not in ("visa_payments", "general_support")):
            escalate = True
            reason = "Payment dispute keyword in non-financial product context — possible misrouted ticket."

        return {
            "escalate": escalate,
            "reason": reason,
            "risk_tier": highest,
            "composite_score": composite,
            "matched_patterns": matched,
        }


# -------------------------
# Module-level helper
# -------------------------
_engine_singleton = None


def should_escalate(text: str, confidence: float = 1.0, product_area: str = "general_support") -> Dict:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = SafetyEngine()
    return _engine_singleton.should_escalate(text, confidence, product_area)