"""
PetroRAG Query Entity Extraction Layer (Module 2.4)
Extracts domain-specific Oil & Gas entities from user queries:
Equipment IDs, Equipment types, Fields, Wells, Assets, Processes,
Failure modes/Conditions, Measurements/Parameters, Units, Standards, Dates, and Safety procedures.
"""

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryEntities(BaseModel):
    """Structured container for domain entities extracted from a query."""
    equipment_ids: List[str] = Field(default_factory=list, description="Specific equipment tags (e.g. S-101, C-101)")
    equipment_types: List[str] = Field(default_factory=list, description="Equipment classifications (e.g. separator, pump)")
    fields: List[str] = Field(default_factory=list, description="Oil/gas field names (e.g. Gullfaks, Permian)")
    wells: List[str] = Field(default_factory=list, description="Well identifiers (e.g. W-12, Well-05)")
    assets: List[str] = Field(default_factory=list, description="Facility or asset names (e.g. Platform Alpha, FPSO)")
    processes: List[str] = Field(default_factory=list, description="Process names (e.g. gas lift, acidizing)")
    parameters: List[str] = Field(default_factory=list, description="Physical parameters (e.g. pressure, vibration, temperature)")
    conditions: List[str] = Field(default_factory=list, description="Operating conditions or failure states (e.g. abnormal/high, cavitation)")
    units: List[str] = Field(default_factory=list, description="Physical measurement units (e.g. bar, psi, MMSCFD)")
    standards: List[str] = Field(default_factory=list, description="Industry engineering standards (e.g. API 610, ASME B31.3)")
    safety_procedures: List[str] = Field(default_factory=list, description="Safety or regulatory procedures (e.g. LOTO, ESD, HAZOP)")
    dates: List[str] = Field(default_factory=list, description="Temporal references or revisions (e.g. 2023, Rev 2)")
    raw_query: str = ""

    def to_metadata_filter(self) -> Dict[str, Any]:
        """Convert extracted entities into Qdrant/PostgreSQL metadata filter criteria."""
        filters: Dict[str, Any] = {}
        if self.equipment_ids:
            filters["equipment_id"] = self.equipment_ids[0]
        if self.equipment_types:
            filters["equipment_type"] = self.equipment_types[0]
        if self.fields:
            filters["field"] = self.fields[0]
        if self.wells:
            filters["well"] = self.wells[0]
        if self.assets:
            filters["asset"] = self.assets[0]
        return filters


class BaseEntityExtractor(ABC):
    """Abstract interface for entity extraction implementations."""

    @abstractmethod
    def extract(self, query: str) -> QueryEntities:
        """Extract domain entities from the query."""
        pass


# ==============================================================================
# GAZETTEERS & PATTERNS
# ==============================================================================

# Equipment ISA/KKS Tag Patterns: e.g. S-101, C-101, P-101A, PSV-402, ESD-01, V-301, TK-501
EQUIPMENT_TAG_REGEX = re.compile(
    r"\b([A-Z]{1,4}-\d{2,4}[A-Z]?(?:/[A-Z])?)\b"
)

# Standards: API, ASME, ISO, NORSOK, NACE
STANDARDS_REGEX = re.compile(
    r"\b(API\s*\d+[A-Z]?|ASME\s*[A-Z0-9.]+|ISO\s*\d+|NORSOK\s*[A-Z0-9-]+|NACE\s*[A-Z0-9]+)\b",
    re.IGNORECASE
)

# Well Patterns: Well 12, W-12, Well-05, Wellhead 4
WELL_REGEX = re.compile(
    r"\b(?:well|wellhead|w)\s*[-#]?\s*([0-9]+[A-Za-z0-9/-]*)\b",
    re.IGNORECASE
)

# Measurement Units: bar, bar(g), psi, psig, bpd, mmscfd, mm/s, rpm, deg C, °C, etc.
UNITS_REGEX = re.compile(
    r"\b(bar\(g\)|barg|psig|bar|psi|kpa|mpa|bpd|mmscfd|scf/d|m3/d|m3/h|gpm|mm/s(?:\s*rms)?|rpm|deg\s*c|°c|deg\s*f|°f|cp|md|ft|meters|ppm)\b",
    re.IGNORECASE
)

# Date / Revision Patterns
DATE_REGEX = re.compile(
    r"\b((?:19|20)\d{2}|rev(?:ision)?\s*\d+|q[1-4]\s*(?:20)?\d{2})\b",
    re.IGNORECASE
)

# Lexicon Gazetteers
EQUIPMENT_TYPES = {
    "compressor": ["compressor", "centrifugal compressor", "reciprocating compressor", "screw compressor"],
    "separator": ["separator", "test separator", "production separator", "2-phase separator", "3-phase separator", "slug catcher"],
    "pump": ["pump", "centrifugal pump", "esp", "electric submersible pump", "positive displacement pump"],
    "valve": ["valve", "control valve", "choke", "choke valve", "psv", "pressure safety valve", "esv", "esd valve", "dump valve", "check valve"],
    "heat_exchanger": ["heat exchanger", "cooler", "reboiler", "condenser"],
    "vessel": ["vessel", "scrubber", "knockout drum", "ko drum", "accumulator"],
    "turbine": ["turbine", "gas turbine", "steam turbine"],
    "wellhead": ["wellhead", "christmas tree", "xmas tree"],
    "bop": ["bop", "blowout preventer"],
    "pipeline": ["pipeline", "flowline", "manifold", "riser"],
}

PARAMETERS = {
    "pressure": ["pressure", "discharge pressure", "suction pressure", "differential pressure", "backpressure"],
    "temperature": ["temperature", "discharge temperature", "exhaust temperature", "skin temperature"],
    "vibration": ["vibration", "radial vibration", "axial vibration", "overall vibration", "displacement"],
    "flow_rate": ["flow rate", "throughput", "production rate", "injection rate", "liquid rate", "gas rate"],
    "level": ["liquid level", "level", "oil-water interface"],
    "water_cut": ["water cut", "bsw", "water-cut"],
    "gor": ["gor", "gas-oil ratio", "gas oil ratio"],
    "porosity": ["porosity"],
    "permeability": ["permeability"],
    "mud_weight": ["mud weight", "fluid density"],
    "rop": ["rate of penetration", "rop"],
}

FAILURE_CONDITIONS = {
    "abnormal/high": ["exceed normal levels", "exceeds normal levels", "high vibration", "high pressure", "high temperature", "overheating", "overpressure", "excessive", "increasing abnormally", "rising abnormally", "abnormal"],
    "abnormal/low": ["pressure drop", "low pressure", "loss of flow", "loss of", "dropping", "depleted"],
    "trip": ["trip", "tripped", "shutdown", "shut down", "interlock trip"],
    "leakage": ["leak", "leaking", "leakage", "loss of containment", "seal failure", "blowout"],
    "cavitation": ["cavitation", "flashing"],
    "surge": ["surge", "surging", "compressor surge"],
    "stuck": ["stuck pipe", "stuck valve", "seized"],
    "corrosion": ["corrosion", "erosion", "wear", "degradation"],
}

KNOWN_FIELDS = [
    "gullfaks", "statfjord", "brent", "ekofisk", "permian", "eagle ford",
    "ghawar", "north sea alpha", "troll", "clair", "mariner", "forties"
]

KNOWN_ASSETS = [
    "platform alpha", "platform bravo", "platform a", "platform b",
    "fpso ocean star", "rig 3", "offshore rig", "gas plant north", "terminal 1"
]

SAFETY_PROCEDURES = {
    "loto": ["loto", "lockout tagout", "lockout/tagout", "isolation procedure"],
    "esd": ["esd", "emergency shutdown", "esd system"],
    "hazop": ["hazop", "hazard and operability"],
    "ptw": ["permit to work", "ptw", "hot work permit", "cold work permit"],
    "ppe": ["ppe", "personal protective equipment", "breathing apparatus"],
    "h2s_safety": ["h2s safety", "toxic gas protocol", "gas detection"],
}

PROCESSES = [
    "gas lift", "water injection", "gas dehydration", "sweetening", "separation",
    "acidizing", "hydraulic fracturing", "fracking", "crude stabilization",
    "flaring", "drilling", "well completion", "well test"
]


class RuleBasedEntityExtractor(BaseEntityExtractor):
    """
    Extracts structured Oil & Gas entities using regular expressions,
    domain gazetteers, and context-dependent condition mappings.
    """

    def extract(self, query: str) -> QueryEntities:
        query_lower = query.lower()

        # 1. Equipment Tag IDs (Preserve original casing)
        eq_ids = EQUIPMENT_TAG_REGEX.findall(query)
        # Filter out common false positives like "API-610" or single letter units
        filtered_eq_ids = [
            tag for tag in eq_ids
            if not tag.upper().startswith(("API-", "ISO-", "REV-"))
        ]

        # 2. Standards
        standards = STANDARDS_REGEX.findall(query)
        normalized_standards = [re.sub(r"\s+", " ", s.upper()) for s in standards]

        # 3. Wells
        well_matches = WELL_REGEX.findall(query)
        wells = [f"Well-{w}" if not w.lower().startswith("w-") else w for w in well_matches]

        # 4. Units
        units = list(set(m.group(0).lower() for m in UNITS_REGEX.finditer(query)))

        # 5. Dates and revisions
        dates = list(set(m.group(0) for m in DATE_REGEX.finditer(query)))

        # 6. Equipment Types
        eq_types = []
        for eq_type, synonyms in EQUIPMENT_TYPES.items():
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", query_lower):
                    eq_types.append(eq_type)
                    break

        # 7. Parameters
        parameters = []
        for param, synonyms in PARAMETERS.items():
            for syn in synonyms:
                if re.search(r"\b" + re.escape(syn) + r"\b", query_lower):
                    parameters.append(param)
                    break

        # 8. Failure Modes / Conditions
        conditions = []
        for cond_label, phrases in FAILURE_CONDITIONS.items():
            for phrase in phrases:
                if phrase in query_lower:
                    conditions.append(cond_label)
                    break

        # Specific prompt condition mapping: "exceed normal levels" -> abnormal/high
        if "exceed normal levels" in query_lower or "exceeded normal levels" in query_lower:
            if "abnormal/high" not in conditions:
                conditions.append("abnormal/high")

        # 9. Fields
        fields = [f.title() for f in KNOWN_FIELDS if re.search(r"\b" + re.escape(f) + r"\b", query_lower)]

        # 10. Assets
        assets = [a.title() for a in KNOWN_ASSETS if re.search(r"\b" + re.escape(a) + r"\b", query_lower)]

        # 11. Safety procedures
        safety = []
        for proc, syns in SAFETY_PROCEDURES.items():
            for syn in syns:
                if re.search(r"\b" + re.escape(syn) + r"\b", query_lower):
                    safety.append(proc.upper())
                    break

        # 12. Processes
        processes = [p for p in PROCESSES if re.search(r"\b" + re.escape(p) + r"\b", query_lower)]

        return QueryEntities(
            equipment_ids=filtered_eq_ids,
            equipment_types=eq_types,
            fields=fields,
            wells=wells,
            assets=assets,
            processes=processes,
            parameters=parameters,
            conditions=conditions,
            units=units,
            standards=normalized_standards,
            safety_procedures=safety,
            dates=dates,
            raw_query=query
        )
