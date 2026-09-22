"""
PetroRAG Operational Database Layer (Module 3.1)
Manages relational database connection (PostgreSQL / SQLite fallback),
SQLAlchemy ORM models for wells, equipment, production time-series,
sensor telemetry, maintenance records, incidents, anomalies, forecasts, and risks.
"""

from datetime import datetime, date
from typing import Generator, Optional, Any, Dict
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Date,
    Text,
    ForeignKey,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from src.core.config import settings
from src.core.logging import logger

Base = declarative_base()


class WellModel(Base):
    """Catalog of oil and gas production and injection wells."""
    __tablename__ = "wells"

    well_id = Column(String(64), primary_key=True)
    well_name = Column(String(128), nullable=False)
    field_name = Column(String(128), nullable=False)
    reservoir = Column(String(128), nullable=True)
    spud_date = Column(Date, nullable=True)
    well_type = Column(String(32), default="PRODUCER")
    status = Column(String(32), default="ACTIVE")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    equipment = relationship("EquipmentModel", back_populates="well", cascade="all, delete-orphan")
    production_records = relationship("ProductionDataModel", back_populates="well", cascade="all, delete-orphan")


class EquipmentModel(Base):
    """Catalog of surface and subsurface equipment assets."""
    __tablename__ = "equipment"

    equipment_id = Column(String(64), primary_key=True)
    well_id = Column(String(64), ForeignKey("wells.well_id"), nullable=True)
    tag_name = Column(String(64), nullable=False, unique=True)
    equipment_type = Column(String(64), nullable=False)
    manufacturer = Column(String(128), nullable=True)
    model = Column(String(128), nullable=True)
    installation_date = Column(Date, nullable=True)
    design_pressure_max_bar = Column(Float, nullable=True)
    design_temp_max_c = Column(Float, nullable=True)
    vibration_alarm_threshold_mms = Column(Float, nullable=True)
    status = Column(String(32), default="OPERATIONAL")

    # Relationships
    well = relationship("WellModel", back_populates="equipment")
    sensor_records = relationship("SensorDataModel", back_populates="equipment", cascade="all, delete-orphan")
    maintenance_records = relationship("MaintenanceModel", back_populates="equipment", cascade="all, delete-orphan")


class ProductionDataModel(Base):
    """Daily/hourly wellhead production time-series."""
    __tablename__ = "production_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    well_id = Column(String(64), ForeignKey("wells.well_id"), nullable=False, index=True)
    oil_rate_bopd = Column(Float, nullable=True)
    gas_rate_mscfd = Column(Float, nullable=True)
    water_rate_bwpd = Column(Float, nullable=True)
    water_cut_pct = Column(Float, nullable=True)
    tubing_head_pressure_psi = Column(Float, nullable=True)
    casing_head_pressure_psi = Column(Float, nullable=True)
    choke_size_64ths = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("well_id", "timestamp", name="uq_well_production_timestamp"),
    )

    well = relationship("WellModel", back_populates="production_records")


class SensorDataModel(Base):
    """High-frequency equipment sensor telemetry time-series."""
    __tablename__ = "equipment_sensor_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    equipment_id = Column(String(64), ForeignKey("equipment.equipment_id"), nullable=False, index=True)
    suction_pressure_bar = Column(Float, nullable=True)
    discharge_pressure_bar = Column(Float, nullable=True)
    temperature_c = Column(Float, nullable=True)
    vibration_rms_mms = Column(Float, nullable=True)
    rpm = Column(Float, nullable=True)
    power_kw = Column(Float, nullable=True)
    flow_rate_m3h = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("equipment_id", "timestamp", name="uq_equipment_sensor_timestamp"),
    )

    equipment = relationship("EquipmentModel", back_populates="sensor_records")


class MaintenanceModel(Base):
    """Equipment maintenance, overhaul, and repair log."""
    __tablename__ = "maintenance_records"

    maintenance_id = Column(String(64), primary_key=True)
    equipment_id = Column(String(64), ForeignKey("equipment.equipment_id"), nullable=False, index=True)
    maintenance_type = Column(String(64), nullable=False)
    date = Column(Date, nullable=False, index=True)
    failure_mode = Column(String(128), nullable=True)
    description = Column(Text, nullable=False)
    resolution = Column(Text, nullable=False)
    parts_replaced = Column(Text, nullable=True)
    downtime_hours = Column(Float, default=0.0)

    equipment = relationship("EquipmentModel", back_populates="maintenance_records")


class IncidentModel(Base):
    """Health, Safety, Environmental, and operational incident reports."""
    __tablename__ = "incidents"

    incident_id = Column(String(64), primary_key=True)
    date = Column(Date, nullable=False, index=True)
    equipment_id = Column(String(64), ForeignKey("equipment.equipment_id"), nullable=True, index=True)
    well_id = Column(String(64), ForeignKey("wells.well_id"), nullable=True, index=True)
    incident_type = Column(String(64), nullable=False)
    severity = Column(String(32), nullable=False)
    description = Column(Text, nullable=False)
    immediate_cause = Column(Text, nullable=True)
    root_cause = Column(Text, nullable=True)
    action_taken = Column(Text, nullable=True)
    document_ref = Column(String(128), nullable=True)


class AnomalyModel(Base):
    """Detected operational anomalies and parameter deviation telemetry."""
    __tablename__ = "anomaly_records"

    anomaly_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    entity_type = Column(String(32), nullable=False)
    entity_id = Column(String(64), nullable=False, index=True)
    detection_method = Column(String(64), nullable=False)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String(32), nullable=False)
    affected_parameter = Column(String(64), nullable=False)
    observed_value = Column(Float, nullable=False)
    baseline_value = Column(Float, nullable=False)
    deviation_pct = Column(Float, nullable=False)
    duration_hours = Column(Float, default=1.0)
    is_verified = Column(Boolean, default=False)
    explanation = Column(Text, nullable=True)


class ForecastModel(Base):
    """Time-series production and pressure forecasts."""
    __tablename__ = "forecast_records"

    forecast_id = Column(String(64), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    entity_id = Column(String(64), nullable=False, index=True)
    target_metric = Column(String(64), nullable=False)
    model_name = Column(String(64), nullable=False)
    horizon_days = Column(Integer, nullable=False)
    forecast_dates = Column(JSON, nullable=False)
    forecast_values = Column(JSON, nullable=False)
    lower_bounds = Column(JSON, nullable=True)
    upper_bounds = Column(JSON, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)


class RiskModel(Base):
    """Operational and process safety risk assessments."""
    __tablename__ = "risk_assessments"

    risk_id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    entity_id = Column(String(64), nullable=False, index=True)
    probability = Column(Integer, nullable=False)
    severity = Column(Integer, nullable=False)
    exposure = Column(Integer, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_level = Column(String(32), nullable=False)
    contributing_factors = Column(JSON, nullable=False)
    mitigation_references = Column(JSON, nullable=False)


# Database Engine & Session Factory
def _build_engine(database_url: Optional[str] = None):
    url = database_url or settings.DATABASE_URL
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, echo=settings.DB_ECHO, connect_args=connect_args)


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(custom_engine=None) -> None:
    """Initialize relational tables."""
    target_engine = custom_engine or engine
    logger.info("Initializing relational database tables...")
    Base.metadata.create_all(bind=target_engine)
    logger.info("Relational database tables successfully created.")


def get_db_session() -> Generator[Session, None, None]:
    """Yield a database session context with automatic rollback and closing."""
    session = SessionLocal()
    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()
