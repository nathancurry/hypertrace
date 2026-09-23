"""Explicit records and structured model responses."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class QuestionStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETE = "complete"


class HypothesisStatus(StrEnum):
    OPEN = "open"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REJECTED = "rejected"


class EpistemicType(StrEnum):
    OBSERVED_USAGE = "observed_usage"
    CHRONOLOGICAL_PRECEDENCE = "chronological_precedence"
    CULTURAL_PROXIMITY = "cultural_proximity"
    POSSIBLE_TRANSMISSION = "possible_transmission"
    ATTRIBUTED_TRANSMISSION = "attributed_transmission"
    DEMONSTRATED_TRANSMISSION = "demonstrated_transmission"


class QuoteRegion(StrEnum):
    ARTICLE_BODY = "article_body"
    TITLE = "title"
    HEADING = "heading"
    NAVIGATION = "navigation"
    TAG = "tag"
    RELATED_CONTENT = "related_content"
    UNKNOWN = "unknown"


class TermSense(StrEnum):
    MODERN_GENRE = "modern_genre"
    EARLIER_UNRELATED_USAGE = "earlier_unrelated_usage"
    SONG_TITLE = "song_title"
    GENERIC_HYPER_MODIFIER = "generic_hyper_modifier"
    RETROSPECTIVE_LABEL = "retrospective_label"
    UNKNOWN = "unknown"


class ContentRegion(BaseModel):
    start: int
    end: int
    region: QuoteRegion
    block_id: int = 0
    target_url: str | None = None


class RelationshipKind(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXTUALIZES = "contextualizes"
    MERELY_PRECEDES = "merely_precedes"
    POSSIBLE_TRANSMISSION = "possible_transmission"
    ATTRIBUTED_TRANSMISSION = "attributed_transmission"
    DEMONSTRATED_TRANSMISSION = "demonstrated_transmission"


class LeadKind(StrEnum):
    PERSON = "person"
    TERM = "term"
    PUBLICATION = "publication"
    DATE = "date"
    CITATION = "citation"
    OTHER = "other"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResearchQuestion(BaseModel):
    id: int | None = None
    question: str = Field(min_length=5)
    created_at: str = Field(default_factory=utc_now)
    status: QuestionStatus = QuestionStatus.ACTIVE


class Hypothesis(BaseModel):
    id: int | None = None
    research_question_id: int
    statement: str = Field(min_length=3)
    status: HypothesisStatus = HypothesisStatus.OPEN
    rationale: str = ""
    created_at: str = Field(default_factory=utc_now)


class SearchQuery(BaseModel):
    id: int | None = None
    research_question_id: int
    query: str = Field(min_length=2, max_length=500)
    rationale: str = ""
    generated_by: str = "model"
    created_at: str = Field(default_factory=utc_now)
    executed_at: str | None = None


class Source(BaseModel):
    id: int | None = None
    canonical_url: str
    retrieved_url: str
    title: str = ""
    author: str | None = None
    page_publication_date: str | None = None
    retrieval_date: str = Field(default_factory=utc_now)
    source_type: str = "web_page"
    archive_url: str | None = None
    content_hash: str
    document_hash: str | None = None
    content: str
    content_regions: list[ContentRegion] = Field(default_factory=list)
    dating_notes: str = ""


class Evidence(BaseModel):
    id: int | None = None
    source_id: int
    research_question_id: int
    exact_quote: str = Field(min_length=1, max_length=1200)
    quote_start: int | None = None
    quote_region: QuoteRegion = QuoteRegion.UNKNOWN
    quote_context: str = ""
    quote_verified_date: str | None = None
    date_verification_note: str = ""
    normalized_claim: str = Field(min_length=1)
    relevance: str = ""
    evidence_type: EpistemicType
    contemporaneous: bool | None = None
    confidence: Confidence = Confidence.LOW
    interpretation_notes: str = ""
    term_sense: TermSense = TermSense.UNKNOWN
    primary_source_verified: bool = False
    transmission_verified: bool = False
    discovered_by_query_id: int | None = None
    created_at: str = Field(default_factory=utc_now)


class Relationship(BaseModel):
    id: int | None = None
    evidence_id: int
    hypothesis_id: int
    kind: RelationshipKind
    rationale: str = ""


class ResearchLead(BaseModel):
    id: int | None = None
    research_question_id: int
    source_id: int
    kind: LeadKind
    value: str = Field(min_length=1, max_length=500)
    rationale: str = ""
    created_at: str = Field(default_factory=utc_now)


class ResearchRun(BaseModel):
    id: int | None = None
    research_question_id: int
    model: str
    provider: str
    started_at: str = Field(default_factory=utc_now)
    ended_at: str | None = None
    status: str = "running"
    actions_taken: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    stop_reason: str = ""


class PlannedQuery(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    rationale: str = Field(min_length=1)


class QueryPlan(BaseModel):
    queries: list[PlannedQuery] = Field(max_length=5)


class ProposedRelationship(BaseModel):
    hypothesis_id: int
    kind: RelationshipKind
    rationale: str = ""


class ExtractedLead(BaseModel):
    kind: LeadKind
    value: str = Field(min_length=1, max_length=500)
    rationale: str = ""


class ExtractedEvidence(BaseModel):
    exact_quote: str = Field(min_length=1, max_length=1200)
    normalized_claim: str = Field(min_length=1)
    relevance: str = ""
    evidence_type: EpistemicType
    contemporaneous: bool | None = None
    confidence: Confidence = Confidence.LOW
    interpretation_notes: str = ""
    term_sense: TermSense = TermSense.UNKNOWN
    hypothesis_links: list[ProposedRelationship] = Field(default_factory=list)


class PageAssessment(BaseModel):
    relevant: bool
    reason: str
    evidence: list[ExtractedEvidence] = Field(default_factory=list, max_length=4)
    leads: list[ExtractedLead] = Field(default_factory=list, max_length=10)
    new_queries: list[PlannedQuery] = Field(default_factory=list, max_length=3)


class HypothesisUpdate(BaseModel):
    hypothesis_id: int
    status: HypothesisStatus
    rationale: str
    evidence_ids: list[int] = Field(default_factory=list)


class Interpretation(BaseModel):
    updates: list[HypothesisUpdate] = Field(default_factory=list)
    new_hypotheses: list[str] = Field(default_factory=list, max_length=2)


class AdversarialReview(BaseModel):
    overclaims: list[str] = Field(default_factory=list, max_length=3)
    dating_concerns: list[str] = Field(default_factory=list, max_length=3)
    source_dependence: list[str] = Field(default_factory=list, max_length=3)
    transmission_gaps: list[str] = Field(default_factory=list, max_length=3)
    falsification_tests: list[str] = Field(default_factory=list, max_length=3)
    next_queries: list[PlannedQuery] = Field(default_factory=list, max_length=5)
