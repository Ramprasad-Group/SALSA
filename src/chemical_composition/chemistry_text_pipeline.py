"""
Sample-centric chemistry text pipeline.

Pipeline stages:
    literature description -> chemistry schema -> canonical chemistry text -> embedding

This module is designed to be:
1) deterministic-first (rule-based baseline),
2) optionally LLM-augmented,
3) robust to underspecified polymer/materials literature,
4) safe for local-only execution (no API keys required).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from llm_client import get_llm_client, load_processing_llm_config
except Exception:  # pragma: no cover - handled by fallback path in runtime
    get_llm_client = None
    load_processing_llm_config = None

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "chem_schema_v1"
CANONICALIZATION_VERSION = "canon_v1"
CANONICAL_TEXT_VERSION = "canon_text_v1"


@dataclass
class SourceSnippet:
    source_type: str
    text: str
    section_label: str = ""
    source_id: str = ""


@dataclass
class SourceMetadata:
    source_type: str
    section_label: str = ""
    source_id: str = ""


@dataclass
class SampleDescriptionBundle:
    sample_id: str
    raw_text: str
    snippets: List[SourceSnippet] = field(default_factory=list)
    source_metadata: List[SourceMetadata] = field(default_factory=list)


@dataclass
class ChemistrySchema:
    sample_name: str
    backbone_family: Optional[str] = None
    polymer_architecture: Optional[str] = None
    repeat_unit_description: Optional[str] = None
    monomer_or_precursor_names: List[str] = field(default_factory=list)
    functionalization_route: List[str] = field(default_factory=list)
    functional_groups: List[str] = field(default_factory=list)
    cation_type: Optional[str] = None
    anion_or_counterion: Optional[str] = None
    crosslinking: Optional[str] = None
    linker_or_spacer: Optional[str] = None
    aromaticity: Optional[str] = None
    fluorination: Optional[str] = None
    hydrophilic_hydrophobic_character: Optional[str] = None
    morphology_cues: List[str] = field(default_factory=list)
    solvent_or_processing_aids: List[str] = field(default_factory=list)
    additive_names: List[str] = field(default_factory=list)
    ion_exchange_capacity_text: Optional[str] = None
    composition_text: Optional[str] = None
    degree_of_functionalization_text: Optional[str] = None
    chemistry_confidence: str = "low"
    ambiguity_notes: List[str] = field(default_factory=list)
    source_support: Dict[str, List[str]] = field(default_factory=dict)
    canonicalization_version: str = ""
    extraction_method: str = "rule_based"
    extraction_model: str = ""
    evidence: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class CanonicalChemistryText:
    sample_id: str
    canonical_text: str
    schema_version: str
    generation_version: str


@dataclass
class ChemistryEmbedding:
    sample_id: str
    canonical_text: str
    embedding_model: str
    embedding_vector: List[float]
    embedding_dim: int
    cache_key: str
    backend: str
    text_hash: str
    timestamp_utc: str


@dataclass
class ChemistryTextPipelineConfig:
    enable_chemistry_text_pipeline: bool = False
    chemistry_schema_extractor: str = "rule_based"  # rule_based | llm | hybrid
    chemistry_embedding_backend: str = "local"  # local | api
    chemistry_embedding_model: str = "hash-embedding-v1"
    exclude_performance_sentences: bool = True
    include_caption_text: bool = True
    include_table_context: bool = True
    include_experimental_section: bool = True


LEAKAGE_PATTERNS = [
    r"\bexcellent conductivity\b",
    r"\bretained\s+\d+(\.\d+)?\s*%.*\b(h|hour)\b",
    r"\bperformed better than\b",
    r"\bpower density\b",
    r"\bpolarization curve\b",
    r"\bstable operation\b",
    r"\bcurrent density\b",
]

BACKBONE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bPEK\b|\bpolyetherketone\b", "polyetherketone"),
    (r"\bpolysulfone\b|\bPSU\b", "polysulfone"),
    (r"\bpoly\(2,6-dimethyl-1,4-phenylene oxide\)\b|\bPPO\b", "poly(2,6-dimethyl-1,4-phenylene oxide)"),
    (r"\bPAES\b|\bpoly\(arylene ether sulfone\)\b", "poly(arylene ether sulfone)"),
]

CATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bquaternary ammonium\b|\bQA\b", "quaternary ammonium"),
    (r"\bimidazolium\b", "imidazolium"),
    (r"\bpiperidinium\b", "piperidinium"),
    (r"\bphosphonium\b", "phosphonium"),
]

ANION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bhydroxide form\b|\bOH-? form\b|\bOH\^\{?-?\}?\b|\bhydroxide\b", "hydroxide"),
    (r"\bchloride form\b|\bCl-? form\b|\bchloride\b", "chloride"),
    (r"\bbicarbonate\b", "bicarbonate"),
]

ARCH_PATTERNS: List[Tuple[str, str]] = [
    (r"\bblock copolymer\b", "block copolymer"),
    (r"\bran(dom)? copolymer\b", "random copolymer"),
    (r"\bcrosslink(ed|ing)\b", "crosslinked"),
]

FUNCTIONALIZATION_MAP: List[Tuple[str, str]] = [
    (r"\bchloromethylat(ed|ion)\b", "chloromethylation"),
    (r"\bquaterniz(ed|ation)\b", "quaternization"),
    (r"\bgraft(ed|ing)\b", "grafting"),
    (r"\balkaliz(ed|ation)\b", "alkalization"),
]

MORPHOLOGY_PATTERNS: List[Tuple[str, str]] = [
    (r"\bphase[-\s]?separated\b", "phase-separated"),
    (r"\bhydrophilic\/hydrophobic\b|\bhydrophilic-hydrophobic\b", "hydrophilic/hydrophobic"),
]

SOLVENT_KEYWORDS = [
    "DMF", "DMSO", "NMP", "DMAc", "THF", "methanol", "ethanol", "water", "toluene"
]

ADDITIVE_HINTS = ["additive", "filler", "dopant", "nanotube", "silica", "graphene", "halloysite"]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def filter_target_leakage_sentences(text: str) -> str:
    """Remove performance-heavy sentences that can leak prediction targets."""
    kept: List[str] = []
    for sentence in _split_sentences(text):
        low = sentence.lower()
        if any(re.search(pat, low) for pat in LEAKAGE_PATTERNS):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def collect_sample_description_text(
    sample: Dict[str, Any],
    parsed_text: str = "",
    tables_text: str = "",
    figures_text: str = "",
    relevant_content: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    include_caption_text: bool = True,
    include_table_context: bool = True,
    include_experimental_section: bool = True,
    exclude_performance_sentences: bool = True,
) -> SampleDescriptionBundle:
    """Collect sample-associated snippets from available paper sources."""
    sample_id = str(sample.get("sample_name", "")).strip() or "unknown_sample"
    snippets: List[SourceSnippet] = []

    def add_snippet(source_type: str, text: str, section_label: str = "", source_id: str = "") -> None:
        txt = (text or "").strip()
        if not txt:
            return
        snippets.append(
            SourceSnippet(
                source_type=source_type,
                text=txt,
                section_label=section_label,
                source_id=source_id,
            )
        )

    # Stage 1: from explicit extracted sample fields
    components = sample.get("components", []) or []
    if components:
        component_desc = "; ".join(
            f"{c.get('name', '')} ({c.get('role', '')}, {c.get('amount', '')})".strip()
            for c in components if isinstance(c, dict)
        )
        add_snippet("sample_components", component_desc, "sample_components")

    for cm in sample.get("co_monomers", []) or []:
        if isinstance(cm, dict):
            add_snippet(
                "sample_co_monomer",
                f"co-monomer {cm.get('name', '')}; ratio {cm.get('ratio', '')}; smiles {cm.get('smiles', '')}",
                "sample_components",
            )

    additive = sample.get("additive", {}) or {}
    if isinstance(additive, dict):
        add_snippet(
            "sample_additive",
            f"additive {additive.get('name', '')}; content {additive.get('content', '')}",
            "sample_components",
        )

    # Stage 2: find sample mentions in parsed text
    sample_pattern = re.escape(sample_id)
    for sent in _split_sentences(parsed_text):
        if re.search(sample_pattern, sent, flags=re.IGNORECASE):
            add_snippet("parsed_text", sent, "body_text")
        elif include_experimental_section and re.search(
            r"\b(experimental|synthesis|prepared|quaterniz|chloromethyl|copolymer)\b",
            sent,
            flags=re.IGNORECASE,
        ):
            add_snippet("parsed_text_context", sent, "experimental_like")

    # Stage 3: table/caption context
    if include_table_context:
        for line in tables_text.splitlines():
            if re.search(sample_pattern, line, flags=re.IGNORECASE):
                add_snippet("table", line, "table_context")
            elif re.search(r"\b(mol%|wt%|ratio|composition|feed)\b", line, flags=re.IGNORECASE):
                add_snippet("table_context", line, "table_context")

    if include_caption_text:
        for line in figures_text.splitlines():
            if re.search(sample_pattern, line, flags=re.IGNORECASE):
                add_snippet("figure_caption", line, "caption")

    # Stage 4: relevant_content snippets
    for item in relevant_content or []:
        if not isinstance(item, dict):
            continue
        excerpt = str(item.get("excerpt", ""))
        sid = str(item.get("source_id", ""))
        if re.search(sample_pattern, excerpt, flags=re.IGNORECASE):
            add_snippet(str(item.get("source_type", "text")), excerpt, "relevant_content", sid)

    raw_text = " ".join(s.text for s in snippets)
    if exclude_performance_sentences:
        raw_text = filter_target_leakage_sentences(raw_text)

    source_metadata = [
        SourceMetadata(source_type=s.source_type, section_label=s.section_label, source_id=s.source_id)
        for s in snippets
    ]
    return SampleDescriptionBundle(
        sample_id=sample_id,
        raw_text=raw_text,
        snippets=snippets,
        source_metadata=source_metadata,
    )


def _collect_pattern_hits(text: str, patterns: List[Tuple[str, str]]) -> Tuple[Optional[str], List[str]]:
    evid: List[str] = []
    for pat, label in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            evid.append(m.group(0))
            return label, evid
    return None, evid


def extract_chemistry_schema_rule_based(
    text: str,
    *,
    sample_name: str,
    extraction_method: str = "rule_based",
) -> ChemistrySchema:
    """Deterministic baseline extractor for AEM/polymer chemistry cues."""
    low = text.lower()
    evidence: Dict[str, List[str]] = {}
    source_support: Dict[str, List[str]] = {}

    backbone, ev = _collect_pattern_hits(text, BACKBONE_PATTERNS)
    if ev:
        evidence["backbone_family"] = ev
        source_support["backbone_family"] = ev

    cation, ev = _collect_pattern_hits(text, CATION_PATTERNS)
    if ev:
        evidence["cation_type"] = ev
        source_support["cation_type"] = ev

    anion, ev = _collect_pattern_hits(text, ANION_PATTERNS)
    if ev:
        evidence["anion_or_counterion"] = ev
        source_support["anion_or_counterion"] = ev

    architecture, ev = _collect_pattern_hits(text, ARCH_PATTERNS)
    if ev:
        evidence["polymer_architecture"] = ev
        source_support["polymer_architecture"] = ev

    route: List[str] = []
    route_evidence: List[str] = []
    for pat, canonical in FUNCTIONALIZATION_MAP:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            route.append(canonical)
            route_evidence.append(m.group(0))
    if route:
        evidence["functionalization_route"] = route_evidence
        source_support["functionalization_route"] = route_evidence

    morphology: List[str] = []
    morph_evidence: List[str] = []
    for pat, canonical in MORPHOLOGY_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            morphology.append(canonical)
            morph_evidence.append(m.group(0))
    if morph_evidence:
        evidence["morphology_cues"] = morph_evidence

    aromaticity = "aromatic" if re.search(r"\baromatic\b", low) else None
    fluorination = "fluorinated" if re.search(r"\bfluoro|perfluoro\b", low) else None
    crosslinking = "crosslinked" if re.search(r"\bcrosslink(ed|ing)\b", low) else None

    solvents = [s for s in SOLVENT_KEYWORDS if re.search(rf"\b{re.escape(s)}\b", text, flags=re.IGNORECASE)]

    additives: List[str] = []
    for sent in _split_sentences(text):
        if any(h in sent.lower() for h in ADDITIVE_HINTS):
            additives.append(sent)

    # Conservative confidence scoring.
    populated = sum(
        v is not None and v != [] and v != ""
        for v in [backbone, architecture, cation, anion, route, aromaticity, morphology]
    )
    confidence = "high" if populated >= 5 else "medium" if populated >= 3 else "low"

    ambiguity_notes: List[str] = []
    if backbone is None:
        ambiguity_notes.append("Backbone family not explicitly specified.")
    if cation is None:
        ambiguity_notes.append("Cation type not explicitly specified.")
    if anion is None:
        ambiguity_notes.append("Counterion not explicitly specified.")

    iec_match = re.search(r"\bIEC[^.]{0,80}", text, flags=re.IGNORECASE)
    comp_match = re.search(r"\b(mol%|wt%|feed ratio|composition)[^.]{0,120}", text, flags=re.IGNORECASE)
    dof_match = re.search(r"\b(degree of functionalization|grafting degree|functionalization degree)[^.]{0,120}", text, flags=re.IGNORECASE)

    monomers = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", text)))[:20]
    functional_groups = sorted(
        set(
            g
            for g in ["quaternary ammonium", "imidazolium", "piperidinium", "chloromethyl", "sulfonyl", "benzyl"]
            if g in low
        )
    )

    return ChemistrySchema(
        sample_name=sample_name,
        backbone_family=backbone,
        polymer_architecture=architecture,
        repeat_unit_description=None,
        monomer_or_precursor_names=monomers,
        functionalization_route=route,
        functional_groups=functional_groups,
        cation_type=cation,
        anion_or_counterion=anion,
        crosslinking=crosslinking,
        linker_or_spacer=None,
        aromaticity=aromaticity,
        fluorination=fluorination,
        hydrophilic_hydrophobic_character=(
            "hydrophilic/hydrophobic"
            if re.search(r"hydrophilic.*hydrophobic|hydrophobic.*hydrophilic", low)
            else None
        ),
        morphology_cues=morphology,
        solvent_or_processing_aids=solvents,
        additive_names=additives,
        ion_exchange_capacity_text=iec_match.group(0) if iec_match else None,
        composition_text=comp_match.group(0) if comp_match else None,
        degree_of_functionalization_text=dof_match.group(0) if dof_match else None,
        chemistry_confidence=confidence,
        ambiguity_notes=ambiguity_notes,
        source_support=source_support,
        canonicalization_version="",
        extraction_method=extraction_method,
        extraction_model="rule_based_regex_v1",
        evidence=evidence,
    )


def extract_chemistry_schema_llm(text: str, *, sample_name: str) -> ChemistrySchema:
    """Optional LLM extractor. Falls back to rule-based on any failure."""
    if get_llm_client is None:
        return extract_chemistry_schema_rule_based(text, sample_name=sample_name, extraction_method="rule_based_fallback")

    prompt = f"""
Extract chemistry schema JSON for sample "{sample_name}" from text.
Be conservative and preserve unknown as null.
Return fields:
sample_name, backbone_family, polymer_architecture, repeat_unit_description,
monomer_or_precursor_names, functionalization_route, functional_groups, cation_type,
anion_or_counterion, crosslinking, linker_or_spacer, aromaticity, fluorination,
hydrophilic_hydrophobic_character, morphology_cues, solvent_or_processing_aids,
additive_names, ion_exchange_capacity_text, composition_text, degree_of_functionalization_text,
chemistry_confidence, ambiguity_notes, source_support, evidence.
Text:
{text}
"""
    try:
        cfg = load_processing_llm_config() if load_processing_llm_config else None
        client = get_llm_client(override=cfg) if cfg else get_llm_client()
        raw, _ = client.chat([{"role": "user", "content": prompt}], temperature=0.0, return_meta=True)
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        schema = extract_chemistry_schema_rule_based(text, sample_name=sample_name, extraction_method="llm_hybrid")
        for key, value in data.items():
            if hasattr(schema, key) and value not in (None, "", [], {}):
                setattr(schema, key, value)
        schema.extraction_method = "llm"
        schema.extraction_model = "processing_llm"
        return schema
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("LLM schema extraction failed for %s: %s", sample_name, exc)
        return extract_chemistry_schema_rule_based(text, sample_name=sample_name, extraction_method="rule_based_fallback")


def canonicalize_chemistry_schema(schema: ChemistrySchema) -> ChemistrySchema:
    """Normalize schema values with AEM/polymer controlled vocabulary."""
    backbone_alias = {
        "pek": "polyetherketone",
        "polyetherketone": "polyetherketone",
        "ppo": "poly(2,6-dimethyl-1,4-phenylene oxide)",
        "poly(2,6-dimethyl-1,4-phenylene oxide)": "poly(2,6-dimethyl-1,4-phenylene oxide)",
        "pek block copolymer": "polyetherketone",
    }
    cation_alias = {"qa": "quaternary ammonium", "quaternary ammonium": "quaternary ammonium"}
    anion_alias = {
        "oh-form": "hydroxide",
        "hydroxide form": "hydroxide",
        "hydroxide": "hydroxide",
        "chloride form": "chloride",
    }

    if schema.backbone_family:
        schema.backbone_family = backbone_alias.get(schema.backbone_family.lower(), schema.backbone_family)
    if schema.cation_type:
        schema.cation_type = cation_alias.get(schema.cation_type.lower(), schema.cation_type)
    if schema.anion_or_counterion:
        schema.anion_or_counterion = anion_alias.get(
            schema.anion_or_counterion.lower(), schema.anion_or_counterion
        )
    if schema.polymer_architecture:
        schema.polymer_architecture = schema.polymer_architecture.lower()
    if schema.aromaticity:
        schema.aromaticity = "aromatic" if "aromatic" in schema.aromaticity.lower() else schema.aromaticity
    if schema.crosslinking:
        schema.crosslinking = "crosslinked" if "cross" in schema.crosslinking.lower() else schema.crosslinking
    schema.canonicalization_version = CANONICALIZATION_VERSION
    return schema


def chemistry_schema_to_text(schema: ChemistrySchema) -> CanonicalChemistryText:
    """Create deterministic chemistry-focused canonical text from schema."""
    sid = schema.sample_name or "Unknown sample"
    backbone = schema.backbone_family or "polymer backbone not fully specified"
    arch = f" {schema.polymer_architecture}" if schema.polymer_architecture else ""
    aromatic_prefix = "fully aromatic " if schema.aromaticity == "aromatic" else ""

    sent1 = f"{sid} is a {aromatic_prefix}{backbone}{arch}."

    route = ", then ".join(schema.functionalization_route) if schema.functionalization_route else "functionalization details are partially unspecified"
    ion = schema.cation_type or "ionic group not fully specified"
    counter = schema.anion_or_counterion or "counterion not fully specified"
    sent2 = f"It was prepared via {route} to form {ion} groups in {counter} form."

    cues: List[str] = []
    if schema.hydrophilic_hydrophobic_character:
        cues.append(schema.hydrophilic_hydrophobic_character)
    if schema.morphology_cues:
        cues.extend(schema.morphology_cues)
    if schema.crosslinking:
        cues.append(schema.crosslinking)
    sent3 = ""
    if cues:
        sent3 = f"The material shows {'; '.join(sorted(set(cues)))} character."

    sent4 = ""
    if schema.ambiguity_notes:
        sent4 = f"Ambiguity: {schema.ambiguity_notes[0]}"

    canonical = " ".join(s for s in [sent1, sent2, sent3, sent4] if s).strip()
    return CanonicalChemistryText(
        sample_id=schema.sample_name,
        canonical_text=canonical,
        schema_version=SCHEMA_VERSION,
        generation_version=CANONICAL_TEXT_VERSION,
    )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_cache_key(text: str, model_name: str, backend: str) -> str:
    return f"{backend}:{model_name}:{_text_hash(text)}"


def _deterministic_hash_embedding(text: str, dim: int = 256) -> np.ndarray:
    """Local fallback embedding with deterministic hashing."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[A-Za-z0-9\-\(\)\/]+", text.lower()):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def load_cached_embedding(cache_dir: Path, cache_key: str) -> Optional[ChemistryEmbedding]:
    meta_path = cache_dir / f"{cache_key}.json"
    vec_path = cache_dir / f"{cache_key}.npy"
    if not meta_path.exists() or not vec_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    vec = np.load(vec_path).astype(np.float32).tolist()
    return ChemistryEmbedding(
        sample_id=meta.get("sample_id", ""),
        canonical_text=meta.get("canonical_text", ""),
        embedding_model=meta.get("embedding_model", ""),
        embedding_vector=vec,
        embedding_dim=int(meta.get("embedding_dim", len(vec))),
        cache_key=cache_key,
        backend=meta.get("backend", "local"),
        text_hash=meta.get("text_hash", ""),
        timestamp_utc=meta.get("timestamp_utc", ""),
    )


def get_text_embedding(
    text: str,
    *,
    sample_id: str,
    model_name: str,
    backend: str,
    cache_dir: Path,
) -> ChemistryEmbedding:
    """Generate embedding using pluggable backend with persistent cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _build_cache_key(text, model_name=model_name, backend=backend)
    cached = load_cached_embedding(cache_dir, cache_key)
    if cached is not None:
        return cached

    vector: np.ndarray
    effective_model = model_name
    if backend == "local":
        if model_name.startswith("sentence-transformers/"):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                model = SentenceTransformer(model_name)
                vector = np.array(model.encode([text])[0], dtype=np.float32)
            except Exception:
                effective_model = "hash-embedding-v1"
                vector = _deterministic_hash_embedding(text)
        else:
            vector = _deterministic_hash_embedding(text)
    elif backend == "api":
        # API backend placeholder. Keep tool functional without network credentials.
        # We intentionally fall back to deterministic local embedding.
        effective_model = f"{model_name}|fallback_hash"
        vector = _deterministic_hash_embedding(text)
    else:
        effective_model = "hash-embedding-v1"
        vector = _deterministic_hash_embedding(text)

    emb = ChemistryEmbedding(
        sample_id=sample_id,
        canonical_text=text,
        embedding_model=effective_model,
        embedding_vector=vector.astype(np.float32).tolist(),
        embedding_dim=int(vector.shape[0]),
        cache_key=cache_key,
        backend=backend,
        text_hash=_text_hash(text),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    np.save(cache_dir / f"{cache_key}.npy", np.array(emb.embedding_vector, dtype=np.float32))
    (cache_dir / f"{cache_key}.json").write_text(json.dumps(asdict(emb), indent=2, ensure_ascii=False))
    return emb


def save_chemistry_text_artifacts(
    base_dir: Path,
    bundle: SampleDescriptionBundle,
    schema: ChemistrySchema,
    canonical: CanonicalChemistryText,
    embedding: Optional[ChemistryEmbedding],
) -> Dict[str, Path]:
    """Persist per-sample artifacts and return path map."""
    base_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    raw_path = base_dir / "raw_description.json"
    raw_path.write_text(json.dumps(asdict(bundle), indent=2, ensure_ascii=False))
    out["raw_description"] = raw_path

    schema_path = base_dir / "chemistry_schema.json"
    schema_path.write_text(json.dumps(asdict(schema), indent=2, ensure_ascii=False))
    out["chemistry_schema"] = schema_path

    canonical_path = base_dir / "canonical_chemistry_text.json"
    canonical_path.write_text(json.dumps(asdict(canonical), indent=2, ensure_ascii=False))
    out["canonical_text"] = canonical_path

    if embedding is not None:
        vec_path = base_dir / "chemistry_embedding.npy"
        np.save(vec_path, np.array(embedding.embedding_vector, dtype=np.float32))
        meta_path = base_dir / "chemistry_embedding_metadata.json"
        meta = asdict(embedding)
        meta.pop("embedding_vector", None)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        out["embedding"] = vec_path
        out["embedding_metadata"] = meta_path
    return out

