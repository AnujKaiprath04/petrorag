"""
Unit Tests for PetroRAG Module 3.15 - Structured Equipment Troubleshooting Workflow
Validates decision tree graph integrity, interactive multi-step progression,
physical safety hold gates (PTW/LOTO), automated dispatch from anomaly alerts,
and conversational RAG grounding context generation.
"""

import pytest
from datetime import datetime

from src.analytics.troubleshooting.workflow import (
    TroubleshootingSession,
    TroubleshootingWorkflowService,
    TroubleshootingResolution,
    DecisionNode,
    DecisionOption,
    build_esp_troubleshooting_tree,
    build_compressor_troubleshooting_tree,
)
from src.analytics.anomaly.explainer import (
    AnomalyExplanationFormatter,
    FactualAnomalyExplanation,
)


def test_decision_tree_graph_integrity():
    """Verify all non-terminal options point to existing nodes and terminal options define root causes."""
    trees = [build_esp_troubleshooting_tree(), build_compressor_troubleshooting_tree()]

    for tree in trees:
        assert len(tree) >= 3
        for node_id, node in tree.items():
            assert node.node_id == node_id
            assert len(node.options) >= 2
            for opt in node.options:
                if opt.is_terminal_root_cause:
                    assert opt.root_cause_tag is not None
                    assert opt.recommended_action is not None
                    assert opt.next_node_id is None
                else:
                    assert opt.next_node_id is not None
                    assert opt.next_node_id in tree, f"Dangling node pointer: {opt.next_node_id} in node {node_id}"


def test_interactive_step_progression_to_terminal_resolution():
    """Test standard multi-step progression from symptom to root-cause diagnosis."""
    session = TroubleshootingSession(
        equipment_id="ESP-WELL-08",
        equipment_type="ESP",
    )

    assert session.is_completed is False
    curr = session.get_current_node()
    assert curr.node_id == "ESP_ROOT"

    # Step 1: Select High Vibration
    ok, msg = session.select_option("VIBRATION_HIGH", operator_notes="RMS spiked to 6.8 mm/s in SCADA.")
    assert ok is True
    assert session.is_completed is False
    assert session.current_node_id == "ESP_VIB_CHECK"
    assert len(session.step_history) == 1

    # Step 2: High temp also observed -> Terminal root cause
    ok, msg = session.select_option("YES_HIGH_TEMP", operator_notes="Thrust bearing RTD reads 91°C.")
    assert ok is True
    assert session.is_completed is True
    assert session.get_current_node() is None

    # Verify resolution
    res = session.resolution
    assert isinstance(res, TroubleshootingResolution)
    assert res.equipment_id == "ESP-WELL-08"
    assert res.total_steps_executed == 2
    assert res.confirmed_root_cause == "JOURNAL_BEARING_DEGRADATION_OR_LUBE_FAILURE"
    assert "shutdown" in res.prescribed_remedial_sop.lower()
    assert res.safety_clearance_type == "MANDATORY_LOTO_LOCKOUT"
    assert res.resolution_status == "RESOLVED"
    assert len(res.audit_trail) == 2


def test_safety_hold_gate_enforcement():
    """Verify operator cannot bypass mandatory safety hold gates without explicit clearance confirmation."""
    session = TroubleshootingSession(
        equipment_id="ESP-WELL-12",
        equipment_type="ESP",
    )

    # Step 1: High Vibration
    session.select_option("VIBRATION_HIGH")
    # Step 2: Normal temperature -> Advances to alignment check
    session.select_option("NO_TEMP_NORMAL")

    curr = session.get_current_node()
    assert curr.node_id == "ESP_ALIGNMENT_CHECK"
    assert curr.safety_hold_gate is not None
    assert "LOTO" in curr.safety_hold_gate

    # Attempt to proceed WITHOUT safety clearance confirmed
    ok_fail, msg_fail = session.select_option("LOOSE_BOLTS_FOUND", safety_confirmed=False)
    assert ok_fail is False
    assert "SAFETY HOLD GATE" in msg_fail
    assert session.is_completed is False

    # Retry WITH safety clearance confirmed
    ok_pass, msg_pass = session.select_option(
        "LOOSE_BOLTS_FOUND",
        operator_notes="LOTO tag applied. Found 2 skid hold-down bolts backed off.",
        safety_confirmed=True,
    )
    assert ok_pass is True
    assert session.is_completed is True
    assert session.resolution.confirmed_root_cause == "MECHANICAL_LOOSENESS_OR_PIPE_STRAIN"


def test_auto_dispatch_from_anomaly_alert():
    """Verify TroubleshootingWorkflowService auto-initializes session from FactualAnomalyExplanation."""
    formatter = AnomalyExplanationFormatter(default_equipment_id="ESP-WELL-14")
    service = TroubleshootingWorkflowService()

    # Cavitation anomaly signature
    explanation = formatter.explain_point(
        features_dict={
            "rpm": 2980.0,
            "motor_current": 49.0,  # Surge
            "discharge_pressure": 8.5,  # Drop
            "suction_pressure": 1.4,
            "bearing_temp": 68.0,
            "vibration_rms": 3.8,
        },
        baseline_medians={
            "rpm": 2980.0,
            "motor_current": 42.0,
            "discharge_pressure": 14.8,
            "suction_pressure": 2.4,
            "bearing_temp": 68.0,
            "vibration_rms": 2.1,
        },
        baseline_iqrs={
            "rpm": 5.0,
            "motor_current": 0.8,
            "discharge_pressure": 0.35,
            "suction_pressure": 0.12,
            "bearing_temp": 0.6,
            "vibration_rms": 0.15,
        },
        anomaly_score=0.85,
        is_anomaly=True,
        equipment_id="ESP-WELL-14",
    )

    session = service.auto_dispatch_from_anomaly(explanation)

    assert isinstance(session, TroubleshootingSession)
    assert session.equipment_id == "ESP-WELL-14"
    assert len(session.step_history) == 1
    assert session.step_history[0].operator_selected_option == "PRESSURE_DROP_CURRENT_SURGE"
    assert session.current_node_id == "ESP_CAVITATION_CHECK"

    # Advance the auto-dispatched session
    ok, _ = session.select_option("GAS_INTERFERENCE", operator_notes="Casing gas vent pressure spiked.")
    assert ok is True
    assert session.is_completed is True
    assert session.resolution.confirmed_root_cause == "WELLBORE_GAS_INTERFERENCE_AND_VAPOR_LOCK"


def test_rag_context_generation():
    """Verify structured workflow grounding context for conversational RAG copilot."""
    session = TroubleshootingSession(
        equipment_id="COMP-01",
        equipment_type="GAS_COMPRESSOR",
    )

    # 1. In-progress context
    ctx_in_progress = session.to_rag_context()
    assert "<TROUBLESHOOTING_WORKFLOW_CONTEXT>" in ctx_in_progress
    assert "</TROUBLESHOOTING_WORKFLOW_CONTEXT>" in ctx_in_progress
    assert "COMP-01" in ctx_in_progress
    assert "[ACTIVE GATE: COMP_ROOT]" in ctx_in_progress
    assert "HIGH_VIBRATION_RADIAL" in ctx_in_progress

    # Advance to completion
    session.select_option("HIGH_VIBRATION_RADIAL")
    session.select_option("SEAL_GAS_CONTAMINATED")

    # 2. Resolved context
    ctx_resolved = session.to_rag_context()
    assert "[SESSION RESOLUTION]" in ctx_resolved
    assert "DRY_GAS_SEAL_FACE_DEGRADATION" in ctx_resolved
    assert "MANDATORY_LOTO_LOCKOUT" in ctx_resolved
