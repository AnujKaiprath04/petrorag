"""
PetroRAG Structured Ingestion Schemas & Alias Mapping (Module 3.2)
Defines standardized dataset types, flexible column alias dictionaries,
and ingestion telemetry models for Oil & Gas structured records.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, date
from pydantic import BaseModel, Field


class DatasetType(str, Enum):
    PRODUCTION = "PRODUCTION"
    EQUIPMENT_SENSOR = "EQUIPMENT_SENSOR"
    EQUIPMENT_CATALOG = "EQUIPMENT_CATALOG"
    MAINTENANCE_RECORD = "MAINTENANCE_RECORD"
    INCIDENT_REPORT = "INCIDENT_REPORT"
    UNKNOWN = "UNKNOWN"


# Industry column synonyms dictionary for robust mapping
COLUMN_ALIASES: Dict[DatasetType, Dict[str, List[str]]] = {
    DatasetType.PRODUCTION: {
        "timestamp": ["timestamp", "date", "time", "datetime", "record_date", "prod_date"],
        "well_id": ["well_id", "well", "well_name", "api_number", "uwi", "well_tag"],
        "oil_rate": ["oil_rate", "oil_rate_bopd", "bopd", "oil_bpd", "q_oil", "oil_prod", "oil_volume"],
        "gas_rate": ["gas_rate", "gas_rate_mscfd", "mscfd", "gas_mscf", "q_gas", "gas_prod", "gas_volume"],
        "water_rate": ["water_rate", "water_rate_bwpd", "bwpd", "water_bpd", "q_water", "water_prod", "water_volume"],
        "water_cut": ["water_cut", "water_cut_pct", "wc_pct", "bsw", "bsw_pct", "water_cut_percentage"],
        "pressure": ["pressure", "tubing_pressure", "thp_psi", "thp", "whp", "tubing_head_pressure_psi", "wellhead_pressure"],
        "temperature": ["temperature", "temp", "wellhead_temp_c", "temperature_c", "tht", "tubing_head_temperature"],
    },
    DatasetType.EQUIPMENT_SENSOR: {
        "timestamp": ["timestamp", "time", "datetime", "date_time", "scan_time", "logged_at"],
        "equipment_id": ["equipment_id", "equipment_tag", "tag", "tag_name", "asset_id", "machine_id"],
        "pressure": ["pressure", "discharge_pressure_bar", "p_discharge", "discharge_press", "p_bar", "discharge_pressure"],
        "temperature": ["temperature", "temp_c", "bearing_temp", "temperature_c", "t_c"],
        "flow_rate": ["flow_rate", "flow_rate_m3h", "flow", "q_m3h", "flowrate"],
        "vibration": ["vibration", "vibration_rms_mms", "vib_rms", "vibration_mms", "vib_overall"],
        "rpm": ["rpm", "speed_rpm", "shaft_rpm", "running_speed", "rotational_speed"],
        "power": ["power", "power_kw", "motor_power", "kw", "power_consumption_kw"],
    },
    DatasetType.EQUIPMENT_CATALOG: {
        "equipment_id": ["equipment_id", "id", "asset_id", "machine_id"],
        "tag_name": ["tag_name", "tag", "equipment_tag"],
        "equipment_type": ["equipment_type", "type", "category", "class"],
        "asset": ["asset", "facility", "plant", "platform", "unit"],
        "field": ["field", "field_name", "area"],
        "well": ["well", "well_id", "associated_well"],
        "manufacturer": ["manufacturer", "oem", "vendor", "make"],
        "model": ["model", "model_number", "series"],
        "status": ["status", "operational_status", "state"],
    },
    DatasetType.MAINTENANCE_RECORD: {
        "maintenance_id": ["maintenance_id", "id", "work_order_id", "wo_number", "wo_id"],
        "equipment_id": ["equipment_id", "equipment_tag", "tag", "asset_id"],
        "maintenance_type": ["maintenance_type", "type", "pm_cm", "category"],
        "date": ["date", "maintenance_date", "performed_date", "completion_date", "timestamp"],
        "failure_mode": ["failure_mode", "fault_code", "symptom", "problem"],
        "description": ["description", "work_description", "issue", "summary"],
        "resolution": ["resolution", "action_taken", "repair_action", "solution"],
    },
    DatasetType.INCIDENT_REPORT: {
        "incident_id": ["incident_id", "id", "report_id", "inc_number"],
        "date": ["date", "incident_date", "event_date", "timestamp"],
        "equipment_id": ["equipment_id", "equipment_tag", "asset_id", "tag"],
        "well_id": ["well_id", "well_name", "well"],
        "incident_type": ["incident_type", "type", "category", "event_type"],
        "severity": ["severity", "severity_level", "risk_level"],
        "description": ["description", "incident_description", "summary", "event_details"],
        "root_cause": ["root_cause", "cause", "underlying_cause", "rca"],
        "action_taken": ["action_taken", "corrective_action", "resolution", "immediate_action"],
    },
}


class IngestionSummary(BaseModel):
    """Execution telemetry for structured ingestion batches."""
    source_name: str
    source_format: str
    dataset_type: DatasetType
    total_rows_read: int
    valid_rows: int
    rejected_rows: int
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    unique_entities: List[str] = Field(default_factory=list)
    persisted_to_database: bool = False
    processing_time_ms: float = 0.0
    errors: List[str] = Field(default_factory=list)
