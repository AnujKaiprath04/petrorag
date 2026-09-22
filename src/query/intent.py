"""
PetroRAG Query Intent Classification Layer (Module 2.3)
Classifies user queries into 15 specialized Oil & Gas intent categories
using a modular hybrid architecture (rule/lexicon scoring with semantic fallback).
"""

from abc import ABC, abstractmethod
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """The 15 canonical Oil & Gas intent categories."""
    TECHNICAL_QA = "TECHNICAL_QA"
    EQUIPMENT = "EQUIPMENT"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    MAINTENANCE = "MAINTENANCE"
    SAFETY = "SAFETY"
    PRODUCTION = "PRODUCTION"
    DRILLING = "DRILLING"
    WELL = "WELL"
    RESERVOIR = "RESERVOIR"
    INCIDENT = "INCIDENT"
    DOCUMENT_SEARCH = "DOCUMENT_SEARCH"
    SUMMARY = "SUMMARY"
    COMPARISON = "COMPARISON"
    ANALYTICS = "ANALYTICS"
    UNKNOWN = "UNKNOWN"


class IntentClassificationResult(BaseModel):
    """Structured intent classification output."""
    primary_intent: QueryIntent
    confidence: float = Field(..., ge=0.0, le=1.0)
    secondary_intents: List[Tuple[QueryIntent, float]] = Field(default_factory=list)
    extracted_clues: Dict[str, Any] = Field(default_factory=dict)
    classifier_used: str = "hybrid"


class BaseIntentClassifier(ABC):
    """Abstract interface for intent classification implementations."""

    @abstractmethod
    def classify(self, query: str) -> IntentClassificationResult:
        """Classify incoming sanitized query into an O&G intent category."""
        pass


# Domain lexicons and regex patterns for each intent
INTENT_PATTERNS: Dict[QueryIntent, Dict[str, Any]] = {
    QueryIntent.TROUBLESHOOTING: {
        "weight": 1.3,
        "keywords": [
            "troubleshoot", "troubleshooting", "trip", "tripped", "vibration", "alarm",
            "overheating", "overheat", "leak", "leaking", "leakage", "cavitation", "surge",
            "surging", "failure", "failed", "abnormal", "abnormally", "defect", "stuck", "fluctuation",
            "pressure drop", "high temperature", "high pressure", "excessive", "loss of",
            "root cause", "unstable", "malfunction", "diagnose", "diagnostic", "rising abnormally",
            "liquid level"
        ],
        "patterns": [
            r"why (is|did|are) .*(increasing|tripping|failing|leaking|vibrating|high|dropping|rising)",
            r"what (caused|causes) .*(trip|alarm|leak|failure|vibration|shut down)",
            r"how to (resolve|fix|diagnose|troubleshoot) .*",
            r".* (abnormal|abnormally|rising abnormally|failing|defect) .*",
        ]
    },
    QueryIntent.MAINTENANCE: {
        "weight": 1.2,
        "keywords": [
            "maintenance", "preventive maintenance", "pm", "lubrication", "lubricate",
            "grease", "bearing replacement", "seal replacement", "mechanical seal", "overhaul",
            "inspection", "service interval", "torque", "calibration", "calibrate", "spare parts",
            "alignment", "rebuild", "schedule", "routine check", "cleaning procedure", "replace"
        ],
        "patterns": [
            r"maintenance (procedure|schedule|interval|frequency|guidelines)",
            r"how (often|to|do i) (inspect|lubricate|service|maintain|overhaul|replace)",
            r"(replacement|replace) (of )?.* (seal|bearing|filter|impeller|gasket|mechanical seal)",
        ]
    },
    QueryIntent.SAFETY: {
        "weight": 1.2,
        "keywords": [
            "safety", "hazard", "hazop", "loto", "lockout", "tagout", "ppe", "h2s",
            "toxic", "esd", "emergency shutdown", "fire", "explosion", "permit to work",
            "ptw", "evacuation", "blast", "flammable", "safety limit", "gas detector",
            "psv", "pressure relief", "asphyxiation", "flaring"
        ],
        "patterns": [
            r"safety (protocol|procedure|precaution|requirement|standard)",
            r"(lockout|tagout|loto) (steps|procedure|isolation)",
            r"emergency (shutdown|response|procedure|action)",
            r"h2s (exposure|safety|threshold|limits)",
        ]
    },
    QueryIntent.DRILLING: {
        "weight": 1.1,
        "keywords": [
            "drilling", "drill", "drill bit", "rop", "rate of penetration", "mud weight",
            "drilling fluid", "drill string", "casing", "cementing", "bop", "blowout preventer",
            "kick", "lost circulation", "mwd", "lwd", "bha", "dogleg", "torque and drag"
        ],
        "patterns": [
            r"drilling (parameters|mud|fluid|operation|problem)",
            r"(mud weight|rop|drill bit) (optimization|calculation|issues)",
            r"wellbore (stability|kick|loss)",
        ]
    },
    QueryIntent.WELL: {
        "weight": 1.2,
        "keywords": [
            "well", "wellhead", "christmas tree", "tubing", "packer", "casing",
            "perforation", "completion", "workover", "acidizing", "fracturing", "fracking",
            "gravel pack", "annulus", "downhole", "choke", "stimulation", "well integrity",
            "production packer"
        ],
        "patterns": [
            r"well (completion|integrity|workover|stimulation|head)",
            r"(tubing|packer|casing) (leak|pressure|installation|depth)",
            r"(procedure for )?(setting|retrieving|run) (a )?.*packer",
            r".*packer at .*",
        ]
    },
    QueryIntent.RESERVOIR: {
        "weight": 1.1,
        "keywords": [
            "reservoir", "porosity", "permeability", "darcy", "skin factor", "depletion",
            "recovery factor", "reserves", "pvt", "bubble point", "aquifer", "coning",
            "material balance", "decline curve", "drive mechanism", "waterflood", "eor"
        ],
        "patterns": [
            r"reservoir (pressure|drive|depletion|fluid|simulation|properties)",
            r"(porosity|permeability|recovery factor) of .*",
            r"(darcy|material balance|pvt) (equation|analysis|data)",
        ]
    },
    QueryIntent.PRODUCTION: {
        "weight": 1.1,
        "keywords": [
            "production", "flow rate", "bpd", "mmscfd", "water cut", "gor", "gas oil ratio",
            "decline", "lift", "gas lift", "esp", "beam pump", "choke size", "separator liquid",
            "crude output", "throughput", "well test", "multiphase"
        ],
        "patterns": [
            r"production (rate|optimization|decline|forecast|loss)",
            r"(water cut|gor|flow rate) (increase|measurement|calculation)",
            r"gas lift (rate|valve|optimization)",
        ]
    },
    QueryIntent.EQUIPMENT: {
        "weight": 1.0,
        "keywords": [
            "compressor", "separator", "pump", "valve", "heat exchanger", "scrubber",
            "cooler", "turbine", "generator", "flare", "slug catcher", "manifold",
            "pipeline", "centrifugal", "reciprocating", "impeller", "casing", "datasheet",
            "specifications", "rating", "model"
        ],
        "patterns": [
            r"what is the (specification|rating|capacity|design|model) of .*",
            r"(compressor|separator|pump|valve) (specifications|datasheet|type)",
        ]
    },
    QueryIntent.INCIDENT: {
        "weight": 1.2,
        "keywords": [
            "incident", "accident", "blowout", "spill", "uncontrolled release", "near miss",
            "investigation", "rca", "root cause analysis", "macondo", "piper alpha",
            "fire incident", "casualty", "injury", "loss of containment"
        ],
        "patterns": [
            r"incident (report|investigation|findings|analysis)",
            r"what happened (during|in) the .* (incident|accident|spill|blowout)",
        ]
    },
    QueryIntent.COMPARISON: {
        "weight": 1.2,
        "keywords": [
            "compare", "comparison", "difference", "differences", "versus", "vs", "vs.",
            "advantages", "disadvantages", "pros and cons", "better than", "distinction"
        ],
        "patterns": [
            r"(difference|differences) between .* and .*",
            r".* (versus|vs|vs\.) .*",
            r"compare .* (with|to|and) .*",
        ]
    },
    QueryIntent.SUMMARY: {
        "weight": 1.2,
        "keywords": [
            "summarize", "summary", "overview", "synopsis", "brief", "key points",
            "executive summary", "abstract", "highlights"
        ],
        "patterns": [
            r"(give|provide|show) (a|an)? (summary|overview) of .*",
            r"summarize (the|this)? .*",
        ]
    },
    QueryIntent.DOCUMENT_SEARCH: {
        "weight": 1.2,
        "keywords": [
            "find document", "search document", "locate", "where is", "manual for",
            "datasheet for", "p&id", "drawing", "sop number", "iso standard", "api standard"
        ],
        "patterns": [
            r"(where|find|locate) (is|the)? (manual|datasheet|p&id|procedure|standard) (for)? .*",
            r"(show|get) me the (document|manual|sop) (for)? .*",
        ]
    },
    QueryIntent.ANALYTICS: {
        "weight": 1.3,
        "keywords": [
            "calculate", "calculation", "formula", "trend", "statistics", "statistical",
            "correlation", "regression", "average", "median", "standard deviation",
            "forecast", "efficiency calculation", "kpi", "compute", "decline curve regression"
        ],
        "patterns": [
            r"how to (calculate|compute) .*",
            r"(calculate|compute|formula for) .*",
            r"(trend|correlation|regression) of .*",
            r".* (regression|regression for|curve regression) .*",
        ]
    },
    QueryIntent.TECHNICAL_QA: {
        "weight": 0.9,
        "keywords": [
            "how does", "what is", "principle", "working principle", "theory",
            "definition", "explain", "describe", "function of"
        ],
        "patterns": [
            r"how does (a|an)? .* work",
            r"what is (the|a|an)? (function|purpose|principle|theory) of .*",
            r"explain (how|what) .*",
        ]
    },
}


class RuleBasedIntentClassifier(BaseIntentClassifier):
    """
    High-precision rule and pattern-based Oil & Gas intent classifier.
    Fast, deterministic, and explains the matched signals.
    """

    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold

    def classify(self, query: str) -> IntentClassificationResult:
        query_lower = query.lower()
        scores: Dict[QueryIntent, float] = {intent: 0.0 for intent in QueryIntent}
        matched_clues: Dict[str, Any] = {}

        for intent, config in INTENT_PATTERNS.items():
            base_weight = config.get("weight", 1.0)
            keywords = config.get("keywords", [])
            patterns = config.get("patterns", [])

            # Keyword matches
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                    scores[intent] += 1.0 * base_weight
                    if intent.value not in matched_clues:
                        matched_clues[intent.value] = []
                    matched_clues[intent.value].append(kw)

            # Regex structural pattern matches
            for pat in patterns:
                if re.search(pat, query_lower):
                    scores[intent] += 2.5 * base_weight
                    if intent.value not in matched_clues:
                        matched_clues[intent.value] = []
                    matched_clues[intent.value].append(f"regex:{pat}")

        # Extract equipment clues
        eq_match = re.search(
            r"\b(compressor|separator|pump|valve|heat exchanger|esp|turbine|choke)\b",
            query_lower
        )
        if eq_match:
            matched_clues["equipment"] = eq_match.group(1)

        # Sort intents by score descending
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_score = sorted_scores[0]

        # Calculate normalized confidence
        if top_score == 0.0:
            return IntentClassificationResult(
                primary_intent=QueryIntent.UNKNOWN,
                confidence=0.1,
                secondary_intents=[],
                extracted_clues=matched_clues,
                classifier_used="rule_based"
            )

        # Compute softmax-like or proportional confidence
        sum_scores = sum(s for _, s in sorted_scores if s > 0)
        confidence = min(0.99, max(0.40, top_score / (sum_scores if sum_scores > 0 else 1.0) * 1.2))

        secondary = [
            (intent, round(score / sum_scores, 3))
            for intent, score in sorted_scores[1:4]
            if score > 0
        ]

        return IntentClassificationResult(
            primary_intent=top_intent,
            confidence=round(confidence, 3),
            secondary_intents=secondary,
            extracted_clues=matched_clues,
            classifier_used="rule_based"
        )


class HybridIntentClassifier(BaseIntentClassifier):
    """
    Hybrid classifier combining fast rule-based classification with
    intelligent fallback logic.
    """

    def __init__(self, rule_classifier: Optional[RuleBasedIntentClassifier] = None):
        self.rule_classifier = rule_classifier or RuleBasedIntentClassifier()

    def classify(self, query: str) -> IntentClassificationResult:
        rule_result = self.rule_classifier.classify(query)

        # If rule classifier is confident, return immediately
        if rule_result.confidence >= 0.45 and rule_result.primary_intent != QueryIntent.UNKNOWN:
            return rule_result

        # Ambiguous query fallback: only map to TECHNICAL_QA if some engineering/industrial signal exists
        if rule_result.primary_intent == QueryIntent.UNKNOWN:
            domain_hints = [
                "oil", "gas", "petroleum", "energy", "fluid", "pressure", "temperature",
                "flow", "rig", "pipe", "pipeline", "mechanism", "device", "process", "system",
                "vessel", "valve", "pump", "compressor", "well", "drilling", "reservoir"
            ]
            has_wh = any(term in query.lower() for term in ["what", "how", "why", "explain", "definition", "principle"])
            has_domain = any(h in query.lower() for h in domain_hints)
            if has_wh and has_domain:
                return IntentClassificationResult(
                    primary_intent=QueryIntent.TECHNICAL_QA,
                    confidence=0.50,
                    secondary_intents=[],
                    extracted_clues={"fallback": "generic_tech_question"},
                    classifier_used="hybrid_fallback"
                )

        return rule_result
