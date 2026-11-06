"""Response shapes for the CAGE ATLAS API (Pydantic)."""
from typing import Optional

from pydantic import BaseModel, ConfigDict


class Loose(BaseModel):
    """Extra fields pass through, so adding a field to the shared package never breaks the API."""
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class ErrorResponse(BaseModel):
    detail: str


class Health(BaseModel):
    status: str


# ---- status -------------------------------------------------------------
class Organization(Loose):
    name: str
    style_model: bool
    data_through: Optional[str] = None
    source: Optional[str] = None


class ModelInfo(Loose):
    groups: Optional[int] = None
    trained_at_utc: Optional[str] = None
    silhouette: Optional[float] = None
    stability_mean_ari: Optional[float] = None


class StatusResponse(Loose):
    data_mode: str
    data_mode_label: str
    organizations: list[Organization]
    profiles_total: int
    model_cohort_size: Optional[int] = None
    model: ModelInfo
    attribution: str
    limits: str


# ---- search, profile ----------------------------------------------------
class SearchHit(Loose):
    fighter_id: str
    fighter_name: Optional[str] = None
    weight_class: Optional[str] = None
    fights: int
    in_model_cohort: bool


class FighterSummary(Loose):
    fighter_id: str
    fighter_name: Optional[str] = None
    weight_class: Optional[str] = None
    most_recent_weight_class: Optional[str] = None
    category: Optional[str] = None
    fights: int
    minutes: float
    first_fight: str
    last_fight: str
    group: str
    in_model_cohort: bool
    undefined_features: list[str]
    evidence_warning: Optional[str] = None


class FeatureValue(Loose):
    feature: str
    label: str
    unit: str
    group: str
    role: str
    higher_means: str
    value: Optional[float] = None
    undefined: bool
    numerator: str
    denominator: str


class FighterProfile(FighterSummary):
    window: str
    features: list[FeatureValue]


# ---- similarity ---------------------------------------------------------
class ExplainItem(Loose):
    feature: str
    label: str
    unit: str
    query_value: Optional[float] = None
    match_value: Optional[float] = None
    std_difference: float


class MatchResult(Loose):
    rank: int
    fighter_id: str
    fighter_name: Optional[str] = None
    weight_class: Optional[str] = None
    most_recent_weight_class: Optional[str] = None
    distance: float
    similarity_index: float
    group: str
    same_group: bool
    fights: int
    minutes: float
    last_fight: str
    undefined_features: list[str]
    closest_characteristics: list[ExplainItem]
    largest_differences: list[ExplainItem]


class Filters(Loose):
    same_cluster: bool
    division: Optional[str] = None
    category: Optional[str] = None
    min_minutes: Optional[float] = None


class CohortInfo(Loose):
    definition: str
    size: int
    candidates_after_filters: int
    filters: Filters
    data_through: Optional[str] = None


class SimilarResponse(Loose):
    query: FighterSummary
    comparison_cohort: CohortInfo
    distance_note: str
    index_note: str
    limits: str
    results: list[MatchResult]
    empty_reason: Optional[str] = None


# ---- style map ----------------------------------------------------------
class MapPoint(Loose):
    fighter_id: str
    fighter_name: Optional[str] = None
    weight_class: Optional[str] = None
    cluster: int
    group: str
    x: float
    y: float


class MapAxes(Loose):
    explained_variance_ratio: list[float]
    explained_variance_total: Optional[float] = None


class MapGroup(Loose):
    cluster: int
    name: str
    size: int


class MapResponse(Loose):
    points: list[MapPoint]
    axes: MapAxes
    groups: list[MapGroup]
    note: str


# ---- comparison ---------------------------------------------------------
class CompareFighter(Loose):
    fighter_id: str
    fighter_name: Optional[str] = None
    weight_class: Optional[str] = None
    category: Optional[str] = None
    fights: int
    minutes: float
    first_fight: str
    last_fight: str
    in_model_cohort: bool
    undefined_features: list[str]
    evidence_warning: Optional[str] = None


class MetricValue(Loose):
    fighter_id: str
    value: Optional[float] = None
    percentile: Optional[float] = None
    undefined: bool


class Metric(Loose):
    feature: str
    label: str
    unit: str
    group: str
    role: str
    higher_means: str
    values: list[MetricValue]


class ReferenceCohort(Loose):
    name: str
    size: int
    definition: str
    percentile_meaning: str


class CompareResponse(Loose):
    fighters: list[CompareFighter]
    metrics: list[Metric]
    reference_cohort: ReferenceCohort
    limits: str


# ---- divisions ----------------------------------------------------------
class DivisionItem(Loose):
    name: str
    fighters: int


class DivisionsResponse(Loose):
    divisions: list[DivisionItem]
    note: str


# ---- past fights --------------------------------------------------------
class FightStats(Loose):
    sig_landed: Optional[float] = None
    sig_attempted: Optional[float] = None
    td_landed: Optional[float] = None
    td_attempted: Optional[float] = None
    kd: Optional[float] = None
    ctrl_sec: Optional[float] = None


class Fight(Loose):
    bout_id: str
    date: str
    opponent_id: str
    opponent_name: Optional[str] = None
    weight_class: Optional[str] = None
    result: str
    method: Optional[str] = None
    round: int
    finish_time: Optional[str] = None
    is_title_fight: bool
    duration_sec: Optional[float] = None
    stats_complete: bool
    fighter: FightStats
    opponent: FightStats
    how_won: str


class FightsFighter(Loose):
    fighter_id: str
    fighter_name: Optional[str] = None


class FightsResponse(Loose):
    fighter: FightsFighter
    total_fights: int
    fights: list[Fight]
    note: str


# ---- latest results and recently active fighters ------------------------
class RecentResult(Loose):
    bout_id: str
    a_id: str
    a_name: Optional[str] = None
    b_id: str
    b_name: Optional[str] = None
    outcome: str
    method: Optional[str] = None
    round: int
    finish_time: Optional[str] = None
    weight_class: Optional[str] = None
    is_title_fight: bool


class RecentFighter(Loose):
    fighter_id: str
    fighter_name: Optional[str] = None
    weight_class: Optional[str] = None
    last_fight: str
    group: str
    fights: int


class RecentResponse(Loose):
    data_through: Optional[str] = None
    latest_event_date: str
    results: list[RecentResult]
    fighters: list[RecentFighter]
    note: str
