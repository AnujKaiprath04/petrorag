"""
Unit Test & Benchmark: Module 2.4 Query Entity Extraction Verification
Measures empirical extraction metrics across O&G entity classes:
- Exact prompt example verification
- Entity parsing: Equipment IDs, Equipment Types, Fields, Wells, Parameters, Conditions, Standards, Units
- Empirical Precision, Recall, and F1 across a manually labeled test set
"""

import pytest
from src.query.entities import RuleBasedEntityExtractor, QueryEntities


@pytest.fixture
def extractor():
    return RuleBasedEntityExtractor()


def test_prompt_example_entity_extraction(extractor):
    """
    Exact prompt requirement:
    "Why did separator S-101 pressure exceed normal levels?"
    Extract:
    equipment_id = S-101
    equipment_type = separator
    parameter = pressure
    condition = abnormal/high
    """
    query = "Why did separator S-101 pressure exceed normal levels?"
    entities: QueryEntities = extractor.extract(query)

    assert "S-101" in entities.equipment_ids
    assert "separator" in entities.equipment_types
    assert "pressure" in entities.parameters
    assert "abnormal/high" in entities.conditions

    # Verify metadata filter conversion for downstream retrieval
    filters = entities.to_metadata_filter()
    assert filters.get("equipment_id") == "S-101"
    assert filters.get("equipment_type") == "separator"


# Manually labeled test dataset for empirical entity extraction benchmark
LABELED_ENTITY_DATASET = [
    {
        "query": "Check API 610 Type BB2 centrifugal pump P-101A discharge pressure at 42.5 bar.",
        "expected": {
            "equipment_ids": ["P-101A"],
            "equipment_types": ["pump"],
            "standards": ["API 610"],
            "parameters": ["pressure"],
            "units": ["bar"],
        }
    },
    {
        "query": "Why did compressor C-101 trip on high vibration in Gullfaks field?",
        "expected": {
            "equipment_ids": ["C-101"],
            "equipment_types": ["compressor"],
            "fields": ["Gullfaks"],
            "parameters": ["vibration"],
            "conditions": ["abnormal/high", "trip"],
        }
    },
    {
        "query": "What is the LOTO isolation protocol for PSV-402 on Platform Alpha?",
        "expected": {
            "equipment_ids": ["PSV-402"],
            "equipment_types": ["valve"],
            "assets": ["Platform Alpha"],
            "safety_procedures": ["LOTO"],
        }
    },
    {
        "query": "Perform matrix acidizing stimulation on Well 12 to reduce skin damage.",
        "expected": {
            "wells": ["Well-12"],
            "processes": ["acidizing"],
        }
    },
    {
        "query": "Monitor gas lift injection rate of 12000 m3/d for production well W-05.",
        "expected": {
            "wells": ["Well-05"],
            "processes": ["gas lift"],
            "units": ["m3/d"],
        }
    },
    {
        "query": "Inspect ASME B31.3 piping manifold for crude oil leak in Statfjord asset.",
        "expected": {
            "standards": ["ASME B31.3"],
            "equipment_types": ["pipeline"],
            "fields": ["Statfjord"],
            "conditions": ["leakage"],
        }
    },
    {
        "query": "What is the emergency shutdown ESD threshold for gas turbine GT-201 in 2023?",
        "expected": {
            "equipment_ids": ["GT-201"],
            "equipment_types": ["turbine"],
            "safety_procedures": ["ESD"],
            "dates": ["2023"],
        }
    },
    {
        "query": "Why did separator S-101 pressure exceed normal levels?",
        "expected": {
            "equipment_ids": ["S-101"],
            "equipment_types": ["separator"],
            "parameters": ["pressure"],
            "conditions": ["abnormal/high"],
        }
    },
]


def test_entity_extraction_empirical_benchmark(extractor):
    """
    Empirical benchmark:
    Measures Precision, Recall, and F1 across all labeled query entities.
    """
    total_expected = 0
    total_extracted = 0
    true_positives = 0

    for item in LABELED_ENTITY_DATASET:
        query = item["query"]
        expected_dict = item["expected"]
        extracted_entities = extractor.extract(query)

        for category, expected_vals in expected_dict.items():
            extracted_vals = getattr(extracted_entities, category, [])
            total_expected += len(expected_vals)
            total_extracted += len(extracted_vals)

            for val in expected_vals:
                if any(val.lower() == str(ext).lower() for ext in extracted_vals):
                    true_positives += 1

    precision = true_positives / total_extracted if total_extracted > 0 else 0.0
    recall = true_positives / total_expected if total_expected > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 60)
    print("EMPIRICAL ENTITY EXTRACTION BENCHMARK")
    print(f"Total Expected Entities:  {total_expected}")
    print(f"Total Extracted Entities: {total_extracted}")
    print(f"True Positive Matches:    {true_positives}")
    print(f"Precision:                {precision:.4f}")
    print(f"Recall:                   {recall:.4f}")
    print(f"F1 Score:                 {f1:.4f}")
    print("=" * 60)

    assert recall >= 0.85, f"Expected entity recall >= 0.85, got {recall:.4f}"
    assert precision >= 0.80, f"Expected entity precision >= 0.80, got {precision:.4f}"
    assert f1 >= 0.82, f"Expected entity F1 >= 0.82, got {f1:.4f}"
