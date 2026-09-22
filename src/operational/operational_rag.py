"""
PetroRAG Operational RAG Service (Module 3.1 & 3.13)
Coordinates ML operational signals, historical incident intelligence,
and Part 2 RAG hybrid retrieval into grounded, explainable operational diagnostics.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.pipeline import PetroRAGPipeline
from backend.app.schemas.rag_response import PetroRAGRequest
from src.analytics.services import (
    IncidentIntelligenceService,
    IncidentSearchRequest,
    RiskAssessmentService,
    RiskAssessmentRequest,
)


class OperationalDiagnosticRequest(BaseModel):
    """Input payload for operational diagnostics combining telemetry and ML signals."""
    entity_id: str
    entity_type: str = "EQUIPMENT"
    symptom_description: str
    observed_telemetry: Dict[str, float] = Field(default_factory=dict)
    anomaly_score: Optional[float] = None
    anomaly_severity: Optional[str] = None
    condition_duration_hours: float = 1.0


class OperationalEvidenceSection(BaseModel):
    document_title: str
    chunk_id: str
    page_number: Optional[int] = None
    excerpt: str
    lineage_hash: Optional[str] = None


class OperationalDiagnosticResponse(BaseModel):
    """Four-part structured operational diagnostic report."""
    entity_id: str
    timestamp: datetime
    observed_condition: str
    ml_signals: Dict[str, Any]
    retrieved_technical_evidence: List[OperationalEvidenceSection]
    historical_similar_incidents: List[Dict[str, Any]]
    recommended_diagnostic_actions: List[str]
    confidence_level: str
    is_grounded: bool
    explanation_statement: str


class OperationalRAGService:
    """
    Connects structured operational signals and ML detections
    to the Part 2 PetroRAG hybrid retrieval and grounding engine.
    """

    def __init__(
        self,
        rag_pipeline: Optional[PetroRAGPipeline] = None,
        incident_service: Optional[IncidentIntelligenceService] = None,
        risk_service: Optional[RiskAssessmentService] = None,
    ):
        self.rag_pipeline = rag_pipeline or PetroRAGPipeline()
        self.incident_service = incident_service or IncidentIntelligenceService()
        self.risk_service = risk_service or RiskAssessmentService()

    async def diagnose_condition(self, request: OperationalDiagnosticRequest) -> OperationalDiagnosticResponse:
        """
        Execute end-to-end operational diagnostic:
        Telemetry + Anomaly -> Technical RAG Retrieval + Incident Similarity -> Explainable Report.
        """
        logger.info(f"Diagnosing operational condition for '{request.entity_id}': {request.symptom_description}")

        # 1. Format Observed Condition
        obs_items = [f"{k}={v:.2f}" for k, v in request.observed_telemetry.items()]
        obs_summary = (
            f"Observed condition on {request.entity_type.lower()} {request.entity_id}: {request.symptom_description}. "
            f"Active telemetry: {', '.join(obs_items) if obs_items else 'Nominal'}."
        )

        # 2. Risk Assessment
        anom_score = request.anomaly_score if request.anomaly_score is not None else 0.5
        risk_res = self.risk_service.assess_risk(
            RiskAssessmentRequest(
                entity_id=request.entity_id,
                active_anomaly_score=anom_score,
                condition_duration_hours=request.condition_duration_hours,
            )
        )

        ml_signals = {
            "anomaly_score": anom_score,
            "anomaly_severity": request.anomaly_severity or "MEDIUM",
            "risk_score": risk_res.risk_score,
            "risk_level": risk_res.risk_level,
            "contributing_factors": risk_res.contributing_factors,
        }

        # 3. Formulate RAG Technical Search Query
        rag_query_text = (
            f"{request.entity_id} {request.symptom_description} operating limits troubleshooting procedure maintenance"
        )
        rag_req = PetroRAGRequest(
            query=rag_query_text,
            top_k=3,
            enable_reranking=True,
            enable_compression=True,
            enable_grounding=True,
        )

        rag_response = await self.rag_pipeline.query(rag_req)

        # Extract Evidence
        evidence_list: List[OperationalEvidenceSection] = []
        for src in rag_response.sources:
            evidence_list.append(
                OperationalEvidenceSection(
                    document_title=src.document_title or "Technical Documentation",
                    chunk_id=src.chunk_id,
                    page_number=src.page_number,
                    excerpt=(src.snippet or "")[:250] + "...",
                    lineage_hash=getattr(src, "lineage_hash", None),
                )
            )

        # 4. Retrieve Historical Similar Incidents
        incident_res = self.incident_service.find_similar_incidents(
            IncidentSearchRequest(
                query_description=request.symptom_description,
                top_k=2,
            )
        )
        similar_incidents = [inc.model_dump() for inc in incident_res.matches]

        # 5. Formulate Diagnostic Actions based on evidence & incidents
        actions = []
        if risk_res.risk_level in ("HIGH", "CRITICAL"):
            actions.append("Initiate immediate field technical inspection and verify operating sensor calibration.")
            actions.append("Review lube oil analysis and check for mechanical wear or seal contamination.")
        else:
            actions.append("Log parameter trend over next 12-hour operational shift.")
            actions.append("Verify operating condition against OEM baseline specification.")

        if similar_incidents:
            actions.append(f"Review corrective action from historical incident {similar_incidents[0]['incident_id']}.")

        # 6. Synthesize Explainable Statement
        explanation = (
            f"[OBSERVED] {obs_summary}\n"
            f"[ML SIGNALS] Anomaly severity: {ml_signals['anomaly_severity']}, Risk: {ml_signals['risk_level']} (Score: {ml_signals['risk_score']}).\n"
            f"[EVIDENCE] {rag_response.answer}\n"
            f"[RECOMMENDED CHECKS] {'; '.join(actions)}"
        )

        is_grounded = not rag_response.abstention.is_abstained and (
            rag_response.grounding.is_grounded if rag_response.grounding else True
        )

        conf_score = rag_response.confidence if isinstance(rag_response.confidence, (int, float)) else 0.85
        if conf_score >= 0.80:
            conf_label = "HIGH"
        elif conf_score >= 0.60:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"

        from datetime import timezone
        return OperationalDiagnosticResponse(
            entity_id=request.entity_id,
            timestamp=datetime.now(timezone.utc),
            observed_condition=obs_summary,
            ml_signals=ml_signals,
            retrieved_technical_evidence=evidence_list,
            historical_similar_incidents=similar_incidents,
            recommended_diagnostic_actions=actions,
            confidence_level=conf_label,
            is_grounded=is_grounded,
            explanation_statement=explanation,
        )

    def diagnose_condition_sync(self, request: OperationalDiagnosticRequest) -> OperationalDiagnosticResponse:
        """Synchronous wrapper for operational diagnosis."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.diagnose_condition(request)).result()
        except RuntimeError:
            return asyncio.run(self.diagnose_condition(request))
