"""
PetroRAG RAG-Enhanced Anomaly Analysis Service (Module 3.13)
Bridges Module 3.12 Factual Anomaly Explanations with Part 2 RAG hybrid retrieval
and historical incident intelligence. Transforms raw anomaly alerts into grounded,
auditable, and safety-compliant operational maintenance directives.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field

from src.core.logging import logger
from src.pipeline import PetroRAGPipeline
from backend.app.schemas.rag_response import PetroRAGRequest
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.analytics.services import (
        IncidentIntelligenceService,
        IncidentSearchRequest,
    )
class OperationalEvidenceSection(BaseModel):
    document_title: str
    chunk_id: str
    page_number: Optional[int] = None
    excerpt: str
    lineage_hash: Optional[str] = None
from src.analytics.anomaly.explainer import (
    FactualAnomalyExplanation,
    AnomalyExplanationFormatter,
    TelemetryObservationSection,
    SafetyEnvelopeSection,
    OperationalRemediationSection,
)


class RAGEnhancedAnomalyReport(BaseModel):
    """
    Comprehensive RAG-grounded operational diagnostic report for field engineering.
    Merges deterministic telemetry observations with retrieved vendor manuals
    and historical incident intelligence.
    """
    report_id: str = Field(default_factory=lambda: f"RAG-ANOM-{uuid.uuid4().hex[:8].upper()}")
    equipment_id: str
    timestamp: str
    factual_explanation: FactualAnomalyExplanation
    retrieved_evidence: List[OperationalEvidenceSection]
    similar_historical_incidents: List[Dict[str, Any]]
    integrated_action_plan: List[str]
    permit_to_work_required: bool
    loto_required: bool = Field(..., description="Lockout/Tagout required before mechanical/electrical intervention")
    grounded_diagnostic_statement: str
    is_grounded: bool
    overall_confidence: float

    def to_executive_summary(self) -> str:
        """One-paragraph high-density executive briefing for offshore superintendents."""
        f_expl = self.factual_explanation
        prim_obs = f_expl.telemetry_observations.primary_driver
        urgency = f_expl.operational_remediation.urgency_level
        failure_mode = f_expl.operational_remediation.diagnosed_failure_mode
        highest_tier = f_expl.safety_envelope.highest_safety_tier

        isolation_str = "MANDATORY LOTO ISOLATION & PERMIT REQUIRED" if self.loto_required else "ROUTINE FIELD ACCESS PERMIT"
        return (
            f"[{self.equipment_id} OPERATIONAL ALERT - {f_expl.overall_severity}] "
            f"Diagnosed {failure_mode} driven by {prim_obs} excursion. "
            f"Safety Envelope: {highest_tier}. Urgency: {urgency} ({isolation_str}). "
            f"Diagnostic grounding confidence: {self.overall_confidence * 100:.1f}%. "
            f"Matched {len(self.similar_historical_incidents)} similar historical events and "
            f"{len(self.retrieved_evidence)} technical references."
        )

    def to_markdown(self) -> str:
        """Formatted operational bulletin in GitHub-flavored Markdown."""
        f_expl = self.factual_explanation
        lines = [
            f"# PetroRAG Operational Intelligence Directive",
            f"**Report ID:** `{self.report_id}` | **Equipment:** `{self.equipment_id}` | **Timestamp:** `{self.timestamp}`",
            f"",
            f"### Executive Diagnostic Summary",
            f"> {self.to_executive_summary()}",
            f"",
            f"### 1. Telemetry Observations & Physical Baselines",
            f"| Sensor Channel | Observed | Baseline Median | Unit | Deviation % | Direction | Intensity (Z) | Driver Role |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for obs in f_expl.telemetry_observations.observations:
            role = "**PRIMARY**" if obs.is_primary_driver else ("Secondary" if obs.is_secondary_driver else "Nominal")
            lines.append(
                f"| `{obs.channel}` | `{obs.observed_value:.2f}` | `{obs.baseline_median:.2f}` | {obs.unit} | "
                f"`{obs.deviation_pct:+.1f}%` | `{obs.direction}` | `{obs.relative_intensity_z:.2f}σ` | {role} |"
            )

        lines.extend([
            f"",
            f"### 2. Safety Standards Compliance",
            f"- **Safety Breach Detected:** `{f_expl.safety_envelope.safety_breach_detected}`",
            f"- **Highest Safety Tier:** `{f_expl.safety_envelope.highest_safety_tier}`",
            f"- **Permit To Work (PTW) Required:** `{self.permit_to_work_required}`",
            f"- **Lockout / Tagout (LOTO) Required:** `{self.loto_required}`",
            f"- **Applicable Standards:** {', '.join(f_expl.safety_envelope.governing_standards) if f_expl.safety_envelope.governing_standards else 'None'}",
            f"- **Assessment:** {f_expl.safety_envelope.safety_summary}",
            f"",
            f"### 3. Retrieved Technical Documentation & OEM Guidance",
        ])

        if self.retrieved_evidence:
            for ev in self.retrieved_evidence:
                page_info = f" (Page {ev.page_number})" if ev.page_number else ""
                lines.append(f"- **{ev.document_title}**{page_info} [`{ev.chunk_id}`]:")
                lines.append(f"  > \"{ev.excerpt.strip()}\"")
        else:
            lines.append("- *No direct OEM documentation retrieved for this specific query.*")

        lines.extend([
            f"",
            f"### 4. Integrated Action & Remediation Plan",
        ])
        for i, step in enumerate(self.integrated_action_plan, 1):
            lines.append(f"{i}. {step}")

        if self.similar_historical_incidents:
            lines.extend([
                f"",
                f"### 5. Historical Precedents & Lessons Learned",
            ])
            for inc in self.similar_historical_incidents:
                lines.append(
                    f"- **{inc.get('incident_id', 'INC-HIST')}** ({inc.get('severity', 'UNKNOWN')}): "
                    f"{inc.get('description', '')[:120]}... Corrective Action: {inc.get('corrective_action', 'N/A')}"
                )

        return "\n".join(lines)


class MockAnomalyRetriever:
    """In-memory fallback retriever for testing and environments with locked vector stores."""

    def __init__(self):
        from src.core.interfaces import RetrievedChunk, RetrievalChannel
        self.sample_chunks = [
            RetrievedChunk(
                chunk_id="chk-oem-esp-manual",
                document_id="DOC-OEM-ESP-01",
                document_title="ESP Operations Manual & Troubleshooting Guide.pdf",
                text="Electric Submersible Pump (ESP) operational limits: Vibration shutdown threshold is 7.1 mm/s RMS (ISO 10816-3 Zone D). If discharge pressure drops while motor current surges, inspect suction strainer for cavitation and gas locking.",
                score=0.96,
                page_number=18,
                section_title="4.1 Operating Envelopes and Alarms",
                channel=RetrievalChannel.HYBRID,
            )
        ]

    def retrieve(self, query: str, top_k: int = 10, filters=None):
        return self.sample_chunks[:top_k]

    async def aretrieve(self, query: str, top_k: int = 10, filters=None):
        return self.sample_chunks[:top_k]


class RAGEnhancedAnomalyService:
    """
    Coordinates anomaly detection signals, factual explanation generation,
    hybrid technical retrieval, and historical knowledge retrieval into
    grounded operational diagnostic reports.
    """

    def __init__(
        self,
        rag_pipeline: Optional[PetroRAGPipeline] = None,
        incident_service: Optional[Any] = None,
        formatter: Optional[AnomalyExplanationFormatter] = None,
    ):
        if incident_service is None:
            from src.analytics.services import IncidentIntelligenceService
            incident_service = IncidentIntelligenceService()

        if rag_pipeline is not None:
            self.rag_pipeline = rag_pipeline
        else:
            try:
                self.rag_pipeline = PetroRAGPipeline()
            except Exception as exc:
                logger.info(f"Default PetroRAGPipeline unavailable ({exc}), initializing in-memory fallback pipeline.")
                from src.generation.llm_provider import MockLLMProvider
                self.rag_pipeline = PetroRAGPipeline(
                    retriever=MockAnomalyRetriever(),
                    llm_provider=MockLLMProvider(
                        default_response="Grounded operational guidance: Follow OEM operating limits and SOP inspection guidelines."
                    ),
                )

        self.incident_service = incident_service
        self.formatter = formatter or AnomalyExplanationFormatter()

    def _formulate_targeted_query(
        self,
        explanation: FactualAnomalyExplanation,
    ) -> str:
        """
        Formulates an information-dense, targeted hybrid retrieval query based on
        diagnosed physical failure mode, primary breached sensor, and safety standards.
        """
        eq_id = explanation.equipment_id
        failure_mode = explanation.operational_remediation.diagnosed_failure_mode.replace("_", " ")
        primary_driver = explanation.telemetry_observations.primary_driver.replace("_", " ")
        standards = " ".join(explanation.safety_envelope.governing_standards)

        # Build clean query terms
        query_parts = [
            eq_id,
            failure_mode,
            f"{primary_driver} troubleshooting procedure",
            standards,
            "maintenance SOP inspection criteria limits",
        ]
        return " ".join(p for p in query_parts if p).strip()

    async def enhance_explanation(
        self,
        explanation: FactualAnomalyExplanation,
        top_k_chunks: int = 3,
    ) -> RAGEnhancedAnomalyReport:
        """
        Enhances a FactualAnomalyExplanation with technical RAG retrieval
        and historical incident matching.
        """
        logger.info(
            f"RAG-enhancing anomaly explanation for {explanation.equipment_id}: "
            f"{explanation.operational_remediation.diagnosed_failure_mode}"
        )

        # 1. Formulate Targeted Search Query
        targeted_query = self._formulate_targeted_query(explanation)

        # 2. Query Hybrid RAG Pipeline with Grounding Block
        rag_prompt_context = explanation.to_rag_context()
        rag_request = PetroRAGRequest(
            query=targeted_query,
            top_k=top_k_chunks,
            enable_reranking=True,
            enable_compression=True,
            enable_grounding=True,
        )

        rag_response = await self.rag_pipeline.query(rag_request)

        # 3. Extract Technical Evidence
        evidence_list: List[OperationalEvidenceSection] = []
        for src in rag_response.sources:
            evidence_list.append(
                OperationalEvidenceSection(
                    document_title=src.document_title or "Technical Operating Manual",
                    chunk_id=src.chunk_id,
                    page_number=src.page_number,
                    excerpt=(src.snippet or "")[:250] + "...",
                    lineage_hash=getattr(src, "lineage_hash", None),
                )
            )

        # 4. Search Historical Incident Knowledge Base
        from src.analytics.services import IncidentSearchRequest
        incident_matches = self.incident_service.find_similar_incidents(
            IncidentSearchRequest(
                query_description=(
                    f"{explanation.operational_remediation.diagnosed_failure_mode} on {explanation.equipment_id} "
                    f"with {explanation.telemetry_observations.primary_driver} excursion"
                ),
                top_k=2,
            )
        )
        historical_incidents = [m.model_dump() for m in incident_matches.matches]

        # 5. Determine Safety Permits and Lockout/Tagout (LOTO)
        highest_tier = explanation.safety_envelope.highest_safety_tier
        urgency = explanation.operational_remediation.urgency_level

        loto_required = (
            highest_tier in ["ZONE_D_TRIP", "CRITICAL_TRIP", "OVERLOAD_TRIP"]
            or urgency == "IMMEDIATE_SHUTDOWN"
        )
        ptw_required = loto_required or (
            highest_tier in ["ZONE_C_ALERT", "WARNING_ALERT", "OVERLOAD_ALERT"]
            or urgency == "URGENT_INSPECTION_24H"
        )

        # 6. Synthesize Integrated Action Plan
        action_plan: List[str] = []

        if loto_required:
            action_plan.append("SAFETY STEP 1: Execute emergency lockout/tagout (LOTO) on main motor breaker and isolate suction/discharge block valves.")
        elif ptw_required:
            action_plan.append("SAFETY STEP 1: Issue Hot Work / Machinery Surveillance Permit to Work (PTW) before field approach.")

        # Add physical SOP recommendations from Module 3.12
        for sop in explanation.operational_remediation.recommended_sop_steps:
            action_plan.append(sop)

        # If historical incidents provided lessons learned, append them
        if historical_incidents:
            top_inc = historical_incidents[0]
            corr = top_inc.get("corrective_action")
            if corr:
                action_plan.append(f"LESSON LEARNED (from {top_inc.get('incident_id', 'historical incident')}): {corr}")

        # 7. Formulate Grounded Diagnostic Statement
        obs_stmt = explanation.telemetry_observations.summary_statement
        rag_answer = rag_response.answer or "Grounded technical guidance applied."
        grounded_stmt = (
            f"[DIAGNOSTIC] {explanation.operational_remediation.diagnosed_failure_mode} diagnosed on {explanation.equipment_id}.\n"
            f"[TELEMETRY] {obs_stmt}\n"
            f"[STANDARDS] {explanation.safety_envelope.safety_summary}\n"
            f"[TECHNICAL GUIDANCE] {rag_answer}"
        )

        is_grounded = not rag_response.abstention.is_abstained and (
            rag_response.grounding.is_grounded if rag_response.grounding else True
        )
        conf_val = rag_response.confidence if isinstance(rag_response.confidence, (int, float)) else 0.85

        return RAGEnhancedAnomalyReport(
            equipment_id=explanation.equipment_id,
            timestamp=explanation.timestamp,
            factual_explanation=explanation,
            retrieved_evidence=evidence_list,
            similar_historical_incidents=historical_incidents,
            integrated_action_plan=action_plan,
            permit_to_work_required=ptw_required,
            loto_required=loto_required,
            grounded_diagnostic_statement=grounded_stmt,
            is_grounded=is_grounded,
            overall_confidence=round(float(conf_val), 3),
        )

    def enhance_explanation_sync(
        self,
        explanation: FactualAnomalyExplanation,
        top_k_chunks: int = 3,
    ) -> RAGEnhancedAnomalyReport:
        """Synchronous wrapper for enhance_explanation."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.enhance_explanation(explanation, top_k_chunks)).result()
        except RuntimeError:
            return asyncio.run(self.enhance_explanation(explanation, top_k_chunks))

    async def analyze_telemetry_point(
        self,
        features_dict: Dict[str, float],
        baseline_medians: Dict[str, float],
        baseline_iqrs: Dict[str, float],
        anomaly_score: float = 0.5,
        is_anomaly: bool = True,
        equipment_id: str = "OFFSHORE_ESP_01",
        ensemble_votes: Optional[Dict[str, bool]] = None,
        top_k_chunks: int = 3,
    ) -> RAGEnhancedAnomalyReport:
        """
        End-to-end telemetry analysis:
        Decomposes telemetry point -> Generates Factual Explanation -> Queries RAG -> Produces Full Report.
        """
        explanation = self.formatter.explain_point(
            features_dict=features_dict,
            baseline_medians=baseline_medians,
            baseline_iqrs=baseline_iqrs,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            equipment_id=equipment_id,
            ensemble_votes=ensemble_votes,
        )

        return await self.enhance_explanation(explanation, top_k_chunks=top_k_chunks)
