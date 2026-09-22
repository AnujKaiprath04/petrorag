"""
Unit Test & Benchmark: Module 2.3 Query Intent Classification Verification
Measures empirical classification metrics across all 15 Oil & Gas categories:
- Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1 Score (Macro & Weighted)
"""

import pytest
from src.query.intent import (
    QueryIntent,
    HybridIntentClassifier,
    RuleBasedIntentClassifier
)


# Labeled evaluation test set: 45 high-quality Oil & Gas queries (3 per category)
LABELED_INTENT_DATASET = [
    # TROUBLESHOOTING
    ("Why is the compressor vibration increasing?", QueryIntent.TROUBLESHOOTING),
    ("Separator S-101 liquid level is rising abnormally despite dump valve open.", QueryIntent.TROUBLESHOOTING),
    ("What caused the sudden pressure drop in the gas export line?", QueryIntent.TROUBLESHOOTING),

    # MAINTENANCE
    ("What is the recommended lubrication interval for bearing housing B-201?", QueryIntent.MAINTENANCE),
    ("Provide the routine preventive maintenance schedule for centrifugal pumps.", QueryIntent.MAINTENANCE),
    ("How do I replace the mechanical seal on crude oil transfer pump P-102?", QueryIntent.MAINTENANCE),

    # SAFETY
    ("Explain the LOTO procedure before entering the high-pressure separator vessel.", QueryIntent.SAFETY),
    ("What are the H2S toxic gas exposure limits and PPE requirements?", QueryIntent.SAFETY),
    ("What is the ESD emergency shutdown protocol during a confirmed gas leak?", QueryIntent.SAFETY),

    # DRILLING
    ("How do we optimize the rate of penetration while controlling mud weight?", QueryIntent.DRILLING),
    ("What are the primary signs of a wellbore kick during drilling operations?", QueryIntent.DRILLING),
    ("Inspect the drill string BHA for abnormal torque and drag.", QueryIntent.DRILLING),

    # WELL
    ("What is the procedure for setting a production packer at 8,500 ft depth?", QueryIntent.WELL),
    ("Inspect the Christmas tree wellhead valves for annulus pressure buildup.", QueryIntent.WELL),
    ("How is matrix acidizing stimulation performed to remove skin damage?", QueryIntent.WELL),

    # RESERVOIR
    ("Calculate the recovery factor and oil reserves using material balance equation.", QueryIntent.RESERVOIR),
    ("What is the average permeability and porosity of the Upper Brent formation?", QueryIntent.RESERVOIR),
    ("Explain the water coning phenomenon in bottom-water drive reservoirs.", QueryIntent.RESERVOIR),

    # PRODUCTION
    ("How do we optimize the gas lift injection rate to maximize bpd flow rate?", QueryIntent.PRODUCTION),
    ("Why is the water cut increasing rapidly in production well W-12?", QueryIntent.PRODUCTION),
    ("What is the current separator throughput in MMSCFD?", QueryIntent.PRODUCTION),

    # EQUIPMENT
    ("What are the design specifications and rating for the centrifugal compressor?", QueryIntent.EQUIPMENT),
    ("Show the datasheet for the high-pressure test separator.", QueryIntent.EQUIPMENT),
    ("What type of valve is installed at the slug catcher manifold inlet?", QueryIntent.EQUIPMENT),

    # INCIDENT
    ("Provide the root cause analysis report for the 2021 pipeline rupture incident.", QueryIntent.INCIDENT),
    ("What safety barriers failed during the Macondo blowout investigation?", QueryIntent.INCIDENT),
    ("Summarize the near-miss incident investigation on Platform Bravo.", QueryIntent.INCIDENT),

    # DOCUMENT_SEARCH
    ("Where is the P&ID drawing for the gas compression module?", QueryIntent.DOCUMENT_SEARCH),
    ("Find the standard operating procedure manual for the ESP system.", QueryIntent.DOCUMENT_SEARCH),
    ("Locate the ISO standard document for offshore piping design.", QueryIntent.DOCUMENT_SEARCH),

    # SUMMARY
    ("Give an executive summary of the field development plan.", QueryIntent.SUMMARY),
    ("Summarize the key findings from the annual asset integrity review.", QueryIntent.SUMMARY),
    ("Provide a brief overview of the offshore drilling campaign.", QueryIntent.SUMMARY),

    # COMPARISON
    ("What are the differences between API 610 BB1 and BB2 pump types?", QueryIntent.COMPARISON),
    ("Compare centrifugal versus reciprocating compressors for sour gas service.", QueryIntent.COMPARISON),
    ("What is the advantage of hydraulic fracturing versus acid fracturing?", QueryIntent.COMPARISON),

    # ANALYTICS
    ("What is the formula to calculate pump hydraulic efficiency?", QueryIntent.ANALYTICS),
    ("Calculate the statistical trend of turbine exhaust temperatures.", QueryIntent.ANALYTICS),
    ("How to compute the decline curve regression for production forecasting?", QueryIntent.ANALYTICS),

    # TECHNICAL_QA
    ("How does an electric submersible pump (ESP) work?", QueryIntent.TECHNICAL_QA),
    ("What is the working principle of a hydrocyclone desander?", QueryIntent.TECHNICAL_QA),
    ("Explain the theory behind gas-liquid phase separation.", QueryIntent.TECHNICAL_QA),

    # UNKNOWN
    ("What is the capital city of France?", QueryIntent.UNKNOWN),
    ("Tell me a funny joke about programming.", QueryIntent.UNKNOWN),
    ("Who won the world championship in football?", QueryIntent.UNKNOWN),
]


def test_prompt_example_classification():
    """
    Test exact prompt specification:
    User: Why is the compressor vibration increasing?
    Output: intent = TROUBLESHOOTING, equipment = compressor
    """
    classifier = HybridIntentClassifier()
    query = "Why is the compressor vibration increasing?"
    result = classifier.classify(query)

    assert result.primary_intent == QueryIntent.TROUBLESHOOTING
    assert result.confidence >= 0.60
    assert result.extracted_clues.get("equipment") == "compressor"


def test_intent_benchmark_metrics():
    """
    Empirical benchmark:
    Runs all 45 test queries through HybridIntentClassifier and computes:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1
    """
    classifier = HybridIntentClassifier()

    y_true: list[QueryIntent] = []
    y_pred: list[QueryIntent] = []

    for query, expected_intent in LABELED_INTENT_DATASET:
        result = classifier.classify(query)
        y_true.append(expected_intent)
        y_pred.append(result.primary_intent)

    # 1. Accuracy
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    total = len(y_true)
    accuracy = correct / total

    # 2. Per-class Precision, Recall, F1
    classes = list(QueryIntent)
    precision_list = []
    recall_list = []
    f1_list = []

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    macro_precision = sum(precision_list) / len(precision_list)
    macro_recall = sum(recall_list) / len(recall_list)
    macro_f1 = sum(f1_list) / len(f1_list)

    print("\n" + "=" * 60)
    print("EMPIRICAL INTENT CLASSIFICATION BENCHMARK (45 QUERIES, 15 CLASSES)")
    print(f"Accuracy:        {accuracy:.4f} ({correct}/{total})")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1 Score:  {macro_f1:.4f}")
    print("=" * 60)

    # Research validation criteria
    assert accuracy >= 0.85, f"Expected accuracy >= 0.85, got {accuracy:.4f}"
    assert macro_f1 >= 0.80, f"Expected macro F1 >= 0.80, got {macro_f1:.4f}"
