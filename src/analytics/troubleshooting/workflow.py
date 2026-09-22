"""
PetroRAG Structured Equipment Troubleshooting Workflow (Module 3.15)
Provides interactive, auditable, multi-step decision trees for oil & gas rotating equipment
and wellhead systems (ESPs, Centrifugal Compressors, Chokes, Pumps).
Combines state machine progression, physical safety hold gates (PTW/LOTO),
operator observation logging, and RAG contextual technical grounding.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal, Union
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.analytics.anomaly.explainer import FactualAnomalyExplanation
from src.analytics.anomaly.rag_enhancer import RAGEnhancedAnomalyReport
from src.analytics.health.engine import ComprehensiveEquipmentHealthReport


# ===========================================================================
# Decision Tree Node Models
# ===========================================================================

class DecisionOption(BaseModel):
    """Selectable answer or observation choice at a troubleshooting gate."""
    option_key: str
    description: str
    next_node_id: Optional[str] = None
    is_terminal_root_cause: bool = False
    root_cause_tag: Optional[str] = None
    recommended_action: Optional[str] = None


class DecisionNode(BaseModel):
    """An operational diagnostic node in the equipment troubleshooting tree."""
    node_id: str
    title: str
    diagnostic_question: str
    physical_parameter_check: Optional[str] = None
    safety_hold_gate: Optional[str] = None
    oem_manual_reference: Optional[str] = None
    options: List[DecisionOption]


class SessionStepRecord(BaseModel):
    """Audit record of an executed troubleshooting decision step."""
    step_number: int
    node_id: str
    node_title: str
    operator_selected_option: str
    operator_notes: Optional[str] = None
    step_timestamp: str
    safety_clearance_confirmed: bool = True


class TroubleshootingResolution(BaseModel):
    """Final diagnostic resolution report upon completing the workflow."""
    session_id: str
    equipment_id: str
    equipment_type: str
    total_steps_executed: int
    confirmed_root_cause: str
    prescribed_remedial_sop: str
    safety_clearance_type: str
    resolution_status: Literal["RESOLVED", "ESCALATED_SHUTDOWN", "IN_PROGRESS", "ABORTED"]
    completed_timestamp: str
    audit_trail: List[SessionStepRecord]


# ===========================================================================
# Built-in Equipment Decision Trees
# ===========================================================================

def build_esp_troubleshooting_tree() -> Dict[str, DecisionNode]:
    """Predefined deterministic decision tree for Electric Submersible Pumps (ESPs)."""
    return {
        "ESP_ROOT": DecisionNode(
            node_id="ESP_ROOT",
            title="Primary Operational Symptom Check",
            diagnostic_question="What is the dominant anomalous operational signature observed on the ESP?",
            physical_parameter_check="Review vibration RMS, motor current, and discharge pressure trends in SCADA.",
            safety_hold_gate=None,
            oem_manual_reference="ESP OEM Field Operation Guide Section 5.1",
            options=[
                DecisionOption(
                    option_key="VIBRATION_HIGH",
                    description="Vibration RMS elevated (>4.5 mm/s Alert or >7.1 mm/s Trip)",
                    next_node_id="ESP_VIB_CHECK",
                ),
                DecisionOption(
                    option_key="PRESSURE_DROP_CURRENT_SURGE",
                    description="Discharge pressure dropping while motor current surges or oscillates",
                    next_node_id="ESP_CAVITATION_CHECK",
                ),
                DecisionOption(
                    option_key="PRESSURE_DROP_CURRENT_DROP",
                    description="Both discharge pressure and motor current dropping sharply",
                    next_node_id="ESP_STARVATION_CHECK",
                ),
                DecisionOption(
                    option_key="MOTOR_TEMP_RUNAWAY",
                    description="Motor stator / bearing temperature rising steadily above 95°C",
                    next_node_id="ESP_THERMAL_CHECK",
                ),
            ],
        ),
        "ESP_VIB_CHECK": DecisionNode(
            node_id="ESP_VIB_CHECK",
            title="Vibration & Thermal Correlation",
            diagnostic_question="Is the high vibration accompanied by elevated bearing temperature (>85°C)?",
            physical_parameter_check="Measure motor thrust bearing temperature RTD sensor.",
            safety_hold_gate="Mandatory Machinery Surveillance Permit if inspecting skid.",
            oem_manual_reference="ISO 10816-3 Mechanical Vibration Severity Chart",
            options=[
                DecisionOption(
                    option_key="YES_HIGH_TEMP",
                    description="Yes, bearing temperature is elevated (>85°C) alongside high vibration",
                    is_terminal_root_cause=True,
                    root_cause_tag="JOURNAL_BEARING_DEGRADATION_OR_LUBE_FAILURE",
                    recommended_action="Execute immediate emergency shutdown. Pull ESP string for bearing disassembly and seal section overhaul.",
                ),
                DecisionOption(
                    option_key="NO_TEMP_NORMAL",
                    description="No, bearing temperature is normal (<75°C); vibration is purely mechanical/acoustic",
                    next_node_id="ESP_ALIGNMENT_CHECK",
                ),
            ],
        ),
        "ESP_ALIGNMENT_CHECK": DecisionNode(
            node_id="ESP_ALIGNMENT_CHECK",
            title="Coupling & Skid Mechanical Integrity",
            diagnostic_question="Perform physical inspection of surface skid, motor flange, and flowline clamping. Are hold-down bolts loose or is pipe strain evident?",
            physical_parameter_check="Torque check on foundation hold-down bolts and dial indicator alignment.",
            safety_hold_gate="LOTO Required before touching rotating drive coupling.",
            oem_manual_reference="API RP 11S3 ESP Installation & Handling",
            options=[
                DecisionOption(
                    option_key="LOOSE_BOLTS_FOUND",
                    description="Loose foundation bolts or misaligned piping flange detected",
                    is_terminal_root_cause=True,
                    root_cause_tag="MECHANICAL_LOOSENESS_OR_PIPE_STRAIN",
                    recommended_action="Re-torque foundation bolts to OEM specification (350 Nm) and adjust piping hangers to eliminate pipe strain.",
                ),
                DecisionOption(
                    option_key="BOLTS_INTACT",
                    description="Skid and foundation bolts are rigid; dynamic rotor unbalance suspected downhole",
                    is_terminal_root_cause=True,
                    root_cause_tag="DOWNHOLE_IMPELLER_FOULING_OR_SHAFT_DEFLECTION",
                    recommended_action="Flush well with hot brine to dissolve scale/wax. If vibration persists, prepare workover rig for pump replacement.",
                ),
            ],
        ),
        "ESP_CAVITATION_CHECK": DecisionNode(
            node_id="ESP_CAVITATION_CHECK",
            title="Hydraulic Suction & Gas Locking Analysis",
            diagnostic_question="Check casing-head gas pressure and suction line differential pressure. Is free gas breakout or intake strainer restriction observed?",
            physical_parameter_check="Verify intake pressure vs bubble-point pressure; check suction strainer differential DP.",
            safety_hold_gate=None,
            oem_manual_reference="API RP 11S4 Sizing and Selection of ESP Systems",
            options=[
                DecisionOption(
                    option_key="GAS_INTERFERENCE",
                    description="Casing pressure high with free gas breakout entering ESP intake (Gas Locking)",
                    is_terminal_root_cause=True,
                    root_cause_tag="WELLBORE_GAS_INTERFERENCE_AND_VAPOR_LOCK",
                    recommended_action="Increase casing-head gas venting to flare/compressor and adjust variable frequency drive (VFD) speed down 10% to stabilize head.",
                ),
                DecisionOption(
                    option_key="STRAINER_CLOG",
                    description="Suction differential pressure high (>1.8 bar) indicating intake screen plugged with sand/scale",
                    is_terminal_root_cause=True,
                    root_cause_tag="SUCTION_INTAKE_SCREEN_PLUGGED",
                    recommended_action="Perform chemical backwash / solvent soak on intake strainer, or execute coil tubing cleanout.",
                ),
            ],
        ),
        "ESP_STARVATION_CHECK": DecisionNode(
            node_id="ESP_STARVATION_CHECK",
            title="Fluid Supply & Mechanical Drive Transmission",
            diagnostic_question="Verify fluid level in wellbore and check motor electrical phase balance. Is well fluid level depleted or is motor running underloaded?",
            physical_parameter_check="Acoustic fluid level survey (echometer) and motor phase ammeter check.",
            safety_hold_gate="High Voltage Safety Clearance for Variable Frequency Drive testing.",
            oem_manual_reference="API RP 11S8 ESP System Vibration & Electrical Testing",
            options=[
                DecisionOption(
                    option_key="WELL_PUMP_OFF",
                    description="Well fluid level is at or below pump intake setting (Pump-Off Condition)",
                    is_terminal_root_cause=True,
                    root_cause_tag="WELLBORE_FLUID_DEPLETION_PUMP_OFF",
                    recommended_action="Throttle surface choke and reduce VFD frequency to match reservoir inflow rate. Set automated underload shutdown.",
                ),
                DecisionOption(
                    option_key="SHAFT_SHEARED",
                    description="Fluid level is high but pump develops zero head with low motor draw (Mechanical Decoupling)",
                    is_terminal_root_cause=True,
                    root_cause_tag="PUMP_SHAFT_SHEAR_OR_COUPLING_FAILURE",
                    recommended_action="Shut down unit immediately. Mobilize workover rig for downhole pump pull and replacement.",
                ),
            ],
        ),
        "ESP_THERMAL_CHECK": DecisionNode(
            node_id="ESP_THERMAL_CHECK",
            title="Cooling Fluid Flow & Motor Electrical Health",
            diagnostic_question="Check fluid velocity past motor shroud and measure motor winding insulation resistance (Megger test). Is fluid velocity sufficient for cooling?",
            physical_parameter_check="Fluid velocity calculation (minimum 1 ft/s past motor) and 1000V Megger test.",
            safety_hold_gate="Lockout/Tagout (LOTO) on MCC breaker prior to Megger insulation testing.",
            oem_manual_reference="IEEE Std 1017 Recommended Practice for Testing ESP Cables",
            options=[
                DecisionOption(
                    option_key="FLOW_INSUFFICIENT",
                    description="Well fluid velocity <1 ft/s or recirculation shroud fouled",
                    is_terminal_root_cause=True,
                    root_cause_tag="INSUFFICIENT_MOTOR_COOLING_FLOW",
                    recommended_action="Clean motor cooling shroud and increase flow rate above critical cooling threshold.",
                ),
                DecisionOption(
                    option_key="INSULATION_BREAKDOWN",
                    description="Megger reading <5 MOhm indicating motor stator insulation degradation",
                    is_terminal_root_cause=True,
                    root_cause_tag="MOTOR_WINDING_INSULATION_BREAKDOWN",
                    recommended_action="Schedule immediate replacement of motor and power cable prior to catastrophic downhole phase-to-ground fault.",
                ),
            ],
        ),
    }


def build_compressor_troubleshooting_tree() -> Dict[str, DecisionNode]:
    """Predefined decision tree for Centrifugal Gas Compressors."""
    return {
        "COMP_ROOT": DecisionNode(
            node_id="COMP_ROOT",
            title="Centrifugal Compressor Primary Disturbance",
            diagnostic_question="What primary operational deviation is triggering alarm on the gas compressor?",
            physical_parameter_check="Review radial vibration, axial displacement, and dry gas seal panel.",
            safety_hold_gate=None,
            oem_manual_reference="API 617 Centrifugal Compressors Section 6",
            options=[
                DecisionOption(
                    option_key="HIGH_VIBRATION_RADIAL",
                    description="Radial vibration exceeding Zone C (4.5 mm/s) or Zone D (7.1 mm/s)",
                    next_node_id="COMP_SEAL_VIB_CHECK",
                ),
                DecisionOption(
                    option_key="SURGE_MARGIN_LOW",
                    description="Anti-surge valve opening frequently; flow operating near surge boundary",
                    next_node_id="COMP_SURGE_CHECK",
                ),
            ],
        ),
        "COMP_SEAL_VIB_CHECK": DecisionNode(
            node_id="COMP_SEAL_VIB_CHECK",
            title="Dry Gas Seal Differential & Lube Oil Condition",
            diagnostic_question="Examine dry gas seal primary vent leakage and seal gas differential pressure. Is seal gas contaminated or leaking?",
            physical_parameter_check="Primary vent flowmeter and seal gas filter delta-P.",
            safety_hold_gate="Gas Detection Clearance Required before inspecting seal panel.",
            oem_manual_reference="API 614 Lubrication and Shaft-Sealing Systems",
            options=[
                DecisionOption(
                    option_key="SEAL_GAS_CONTAMINATED",
                    description="High primary vent leakage (>100 SCFM) and black oil particulate in seal drain",
                    is_terminal_root_cause=True,
                    root_cause_tag="DRY_GAS_SEAL_FACE_DEGRADATION_AND_LUBE_MIGRATION",
                    recommended_action="Execute controlled compressor train shutdown. Isolate gas loop with blinds and replace dry gas seal cartridge.",
                ),
                DecisionOption(
                    option_key="SEAL_INTACT_BEARING_ISSUE",
                    description="Seal gas parameters nominal; vibration frequency peaks at 1X or 0.43X (oil whirl)",
                    is_terminal_root_cause=True,
                    root_cause_tag="HYDRODYNAMIC_JOURNAL_BEARING_OIL_WHIRL",
                    recommended_action="Increase lube oil inlet temperature slightly to reduce viscosity, inspect tilt-pad bearing clearance at earliest turnaround.",
                ),
            ],
        ),
        "COMP_SURGE_CHECK": DecisionNode(
            node_id="COMP_SURGE_CHECK",
            title="Process Suction Conditions & Molecular Weight Shift",
            diagnostic_question="Verify suction scrubber level and gas chromatography molecular weight (MW). Did heavy hydrocarbon liquid carryover occur?",
            physical_parameter_check="Suction drum liquid level transmitters and gas MW analyzer.",
            safety_hold_gate=None,
            oem_manual_reference="API 617 Anti-Surge Control Guidelines",
            options=[
                DecisionOption(
                    option_key="LIQUID_CARRYOVER",
                    description="High liquid level in suction scrubber with droplet carryover into impeller",
                    is_terminal_root_cause=True,
                    root_cause_tag="SUCTION_SCRUBBER_LIQUID_CARRYOVER",
                    recommended_action="Open scrubber automated drain valve immediately, blow down demister pad, and reset anti-surge safety bias.",
                ),
                DecisionOption(
                    option_key="MOLECULAR_WEIGHT_DROP",
                    description="Gas composition shifted to lighter gas (methane surge) moving surge line to the right",
                    is_terminal_root_cause=True,
                    root_cause_tag="GAS_MOLECULAR_WEIGHT_COMPOSITION_SHIFT",
                    recommended_action="Update anti-surge controller molecular weight compensation algorithm to restore 15% surge margin safety envelope.",
                ),
            ],
        ),
    }


# ===========================================================================
# Interactive Troubleshooting Session State Machine
# ===========================================================================

class TroubleshootingSession:
    """
    Manages an active, stateful, multi-step troubleshooting session for an oil & gas asset.
    Maintains audit trail, checks safety gates, advances decision nodes, and outputs
    grounded diagnostic reports.
    """

    def __init__(
        self,
        equipment_id: str,
        equipment_type: str = "ESP",
        tree: Optional[Dict[str, DecisionNode]] = None,
        session_id: Optional[str] = None,
        anomaly_explanation: Optional[FactualAnomalyExplanation] = None,
        health_report: Optional[ComprehensiveEquipmentHealthReport] = None,
    ):
        self.session_id = session_id or f"TRBL-{uuid.uuid4().hex[:8].upper()}"
        self.equipment_id = equipment_id
        self.equipment_type = equipment_type.upper()
        self.tree = tree or self._load_default_tree(self.equipment_type)
        self.root_node_id = f"{self.equipment_type[:3]}_ROOT" if f"{self.equipment_type[:3]}_ROOT" in self.tree else list(self.tree.keys())[0]
        self.current_node_id: Optional[str] = self.root_node_id
        self.step_history: List[SessionStepRecord] = []
        self.is_completed = False
        self.resolution: Optional[TroubleshootingResolution] = None
        self.anomaly_explanation = anomaly_explanation
        self.health_report = health_report
        self.created_at = datetime.now(timezone.utc).isoformat()

    def _load_default_tree(self, eq_type: str) -> Dict[str, DecisionNode]:
        if "COMPRESSOR" in eq_type:
            return build_compressor_troubleshooting_tree()
        return build_esp_troubleshooting_tree()

    def get_current_node(self) -> Optional[DecisionNode]:
        """Returns the current active decision node or None if session is complete."""
        if self.is_completed or self.current_node_id is None:
            return None
        return self.tree.get(self.current_node_id)

    def select_option(
        self,
        option_key: str,
        operator_notes: Optional[str] = None,
        safety_confirmed: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Executes an operator decision step:
        Validates safety gates, transitions to next node or resolves session.
        Returns (success, message).
        """
        if self.is_completed:
            return False, f"Session {self.session_id} is already completed."

        node = self.get_current_node()
        if not node:
            return False, "No active node in session."

        if node.safety_hold_gate and not safety_confirmed:
            return False, f"SAFETY HOLD GATE: {node.safety_hold_gate}. Operator must confirm clearance before proceeding."

        # Find matching option
        match_opt = next((o for o in node.options if o.option_key == option_key), None)
        if not match_opt:
            valid_keys = [o.option_key for o in node.options]
            return False, f"Invalid option '{option_key}'. Must be one of: {valid_keys}"

        # Record step
        step_rec = SessionStepRecord(
            step_number=len(self.step_history) + 1,
            node_id=node.node_id,
            node_title=node.title,
            operator_selected_option=option_key,
            operator_notes=operator_notes,
            step_timestamp=datetime.now(timezone.utc).isoformat(),
            safety_clearance_confirmed=safety_confirmed,
        )
        self.step_history.append(step_rec)

        # Handle terminal root cause
        if match_opt.is_terminal_root_cause:
            self.is_completed = True
            self.current_node_id = None

            # Determine safety clearance requirements
            root_tag = match_opt.root_cause_tag or "UNSPECIFIED_FAILURE_MODE"
            is_emergency = "SHUTDOWN" in (match_opt.recommended_action or "").upper()
            safety_type = "MANDATORY_LOTO_LOCKOUT" if is_emergency else "PERMIT_TO_WORK_CLEARANCE"

            self.resolution = TroubleshootingResolution(
                session_id=self.session_id,
                equipment_id=self.equipment_id,
                equipment_type=self.equipment_type,
                total_steps_executed=len(self.step_history),
                confirmed_root_cause=root_tag,
                prescribed_remedial_sop=match_opt.recommended_action or "Execute standard maintenance protocol.",
                safety_clearance_type=safety_type,
                resolution_status="RESOLVED",
                completed_timestamp=datetime.now(timezone.utc).isoformat(),
                audit_trail=self.step_history,
            )
            return True, f"Root cause confirmed: {root_tag}. Remedial action prescribed."

        # Advance to next node
        if match_opt.next_node_id and match_opt.next_node_id in self.tree:
            self.current_node_id = match_opt.next_node_id
            return True, f"Advanced to node: {self.current_node_id} ({self.tree[self.current_node_id].title})"
        else:
            self.is_completed = True
            return False, f"Next node '{match_opt.next_node_id}' not found in tree. Session aborted."

    def to_rag_context(self) -> str:
        """
        Formats active session progress or final resolution into structured
        grounding context for LLM copilot dialog.
        """
        history_lines = [
            f"Step {s.step_number} [{s.node_id}]: Selected '{s.operator_selected_option}' "
            f"(Notes: {s.operator_notes or 'None'})"
            for s in self.step_history
        ]

        if self.resolution:
            res_block = f"""[SESSION RESOLUTION]
Confirmed Root Cause: {self.resolution.confirmed_root_cause}
Prescribed Action: {self.resolution.prescribed_remedial_sop}
Safety Clearance Required: {self.resolution.safety_clearance_type}
Status: {self.resolution.resolution_status}"""
        else:
            curr = self.get_current_node()
            opt_str = "\n".join(f"- {o.option_key}: {o.description}" for o in curr.options) if curr else "None"
            res_block = f"""[ACTIVE GATE: {curr.node_id if curr else 'None'}]
Question: {curr.diagnostic_question if curr else 'None'}
Check: {curr.physical_parameter_check if curr else 'None'}
Safety Gate: {curr.safety_hold_gate if curr else 'None'}
Available Choices:
{opt_str}"""

        return f"""<TROUBLESHOOTING_WORKFLOW_CONTEXT>
Session ID: {self.session_id}
Equipment: {self.equipment_id} ({self.equipment_type})
Completed: {self.is_completed}

[EXECUTION AUDIT TRAIL]
{chr(10).join(history_lines) if history_lines else 'No steps executed yet.'}

{res_block}
</TROUBLESHOOTING_WORKFLOW_CONTEXT>"""


# ===========================================================================
# Automated Workflow Dispatcher Service
# ===========================================================================

class TroubleshootingWorkflowService:
    """
    Manages active troubleshooting sessions across the facility,
    auto-initializes sessions from RAG anomaly alerts, and produces final RCA documentation.
    """

    def __init__(self):
        self.active_sessions: Dict[str, TroubleshootingSession] = {}

    def start_session(
        self,
        equipment_id: str,
        equipment_type: str = "ESP",
        anomaly_explanation: Optional[FactualAnomalyExplanation] = None,
        health_report: Optional[ComprehensiveEquipmentHealthReport] = None,
    ) -> TroubleshootingSession:
        """Initializes a new interactive diagnostic session."""
        session = TroubleshootingSession(
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            anomaly_explanation=anomaly_explanation,
            health_report=health_report,
        )
        self.active_sessions[session.session_id] = session
        logger.info(f"Started troubleshooting session {session.session_id} for {equipment_id} ({equipment_type})")
        return session

    def get_session(self, session_id: str) -> Optional[TroubleshootingSession]:
        return self.active_sessions.get(session_id)

    def auto_dispatch_from_anomaly(
        self,
        explanation: FactualAnomalyExplanation,
    ) -> TroubleshootingSession:
        """
        Auto-initializes a troubleshooting session and pre-selects the first gate
        based on the detected physical failure mode.
        """
        session = self.start_session(
            equipment_id=explanation.equipment_id,
            equipment_type="ESP",
            anomaly_explanation=explanation,
        )

        # Pre-select matching initial symptom based on failure mode
        mode = explanation.operational_remediation.diagnosed_failure_mode
        if "VIBRATION" in mode or "UNBALANCE" in mode or "BEARING" in mode:
            session.select_option("VIBRATION_HIGH", operator_notes="Auto-selected from telemetry vibration excursion.")
        elif "CAVITATION" in mode:
            session.select_option("PRESSURE_DROP_CURRENT_SURGE", operator_notes="Auto-selected from telemetry P-I cavitation signature.")
        elif "STARVATION" in mode:
            session.select_option("PRESSURE_DROP_CURRENT_DROP", operator_notes="Auto-selected from telemetry pressure/current drop.")
        elif "THERMAL" in mode:
            session.select_option("MOTOR_TEMP_RUNAWAY", operator_notes="Auto-selected from telemetry bearing thermal runaway.")

        return session
