"""CAGE ATLAS web API: a thin layer over the shared analysis package.

No model is trained and no raw data is read while handling a request; the fitted model
and fighter profiles are loaded once and kept in memory.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analysis.cageatlas import service
from analysis.cageatlas.similarity import (
    FighterNotFound,
    find_similar,
    load_atlas,
    search_fighters,
)

from .schemas import (
    CompareResponse,
    DivisionsResponse,
    ErrorResponse,
    FighterProfile,
    FightsResponse,
    Health,
    MapResponse,
    RecentResponse,
    SearchHit,
    SimilarResponse,
    StatusResponse,
)

log = logging.getLogger("cageatlas")

FIGHTER_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
_FIGHTER_ID = re.compile(FIGHTER_ID_PATTERN)
UNAVAILABLE = ("The style model or fighter profiles could not be loaded. Build them with "
               "the data and training commands in the README, then retry.")
_load_lock = threading.Lock()


def _try_load(app):
    """Load the model and profiles once. Failure is recorded, not raised, so the server
    keeps running and can report 'source unavailable' (and recover on a later retry)."""
    with _load_lock:
        if getattr(app.state, "atlas", None) is not None:
            return
        try:
            app.state.atlas = load_atlas()
            app.state.atlas_error = None
        except Exception as exc:  # noqa: BLE001 - reported to the caller as a 503
            log.warning("could not load atlas: %s: %s", type(exc).__name__, exc)
            app.state.atlas = None
            app.state.atlas_error = type(exc).__name__


@asynccontextmanager
async def lifespan(app):
    app.state.atlas = None
    app.state.atlas_error = None
    _try_load(app)
    yield


app = FastAPI(
    title="CAGE ATLAS API",
    description="Fighter style similarity. A statistical comparison tool, not a prediction tool.",
    version="0.1.0",
    lifespan=lifespan,
)

_origins = os.environ.get("CAGE_ATLAS_CORS_ORIGINS",
                          "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_atlas(request: Request):
    app_ = request.app
    if getattr(app_.state, "atlas", None) is None:
        _try_load(app_)                                  # retry on request
    atlas = getattr(app_.state, "atlas", None)
    if atlas is None:
        kind = getattr(app_.state, "atlas_error", None)
        raise HTTPException(status_code=503,
                            detail=UNAVAILABLE + (f" ({kind})" if kind else ""))
    return atlas


def get_bouts(request: Request):
    """Past-fight tables load on first use, so a missing file only affects Replay."""
    state = request.app.state
    if getattr(state, "bouts", None) is None:
        try:
            state.bouts = service.load_bouts()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not load bouts: %s: %s", type(exc).__name__, exc)
            raise HTTPException(503, "Past-fight data could not be loaded. Build it with the "
                                     "data commands in the README, then retry.")
    return state.bouts


@app.exception_handler(FighterNotFound)
async def fighter_not_found(request: Request, exc: FighterNotFound):
    return JSONResponse(status_code=404,
                        content={"detail": f"No fighter found with id '{exc.args[0]}'."})


def _cached(request, atlas, key, build):
    """Cache read-only responses per loaded atlas (the reference keeps the id unique)."""
    cache = getattr(request.app.state, "cache", None)
    if cache is None or cache.get("atlas") is not atlas:
        cache = {"atlas": atlas}
        request.app.state.cache = cache
    if key not in cache:
        cache[key] = build(atlas)
    return cache[key]


def _check_ids(ids):
    if not 2 <= len(ids) <= 4 or len(set(ids)) != len(ids):
        raise HTTPException(422, "Choose 2 to 4 different fighters (ids=...&ids=...).")
    if not all(_FIGHTER_ID.match(i) for i in ids):
        raise HTTPException(422, "Fighter ids may only contain letters, digits, '_' and '-'.")


NOT_FOUND = {404: {"model": ErrorResponse}}
UNAVAILABLE_RESPONSE = {503: {"model": ErrorResponse}}


@app.get("/api/health", response_model=Health)
def health():
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse, responses=UNAVAILABLE_RESPONSE)
def status(atlas=Depends(get_atlas)):
    return service.dataset_status(atlas)


@app.get("/api/divisions", response_model=DivisionsResponse, responses=UNAVAILABLE_RESPONSE)
def divisions(request: Request, atlas=Depends(get_atlas)):
    def build(a):
        col = "primary_weight_class" if "primary_weight_class" in a.profiles else "weight_class"
        counts = a.profiles[a.in_cohort][col].dropna().value_counts()
        return {
            "divisions": [{"name": n, "fighters": int(c)} for n, c in sorted(counts.items())],
            "note": "Counts are fighters in the model cohort, by the division they fought in most.",
        }
    return _cached(request, atlas, "divisions", build)


@app.get("/api/fighters/search", response_model=list[SearchHit],
         responses=UNAVAILABLE_RESPONSE)
def search(q: str = Query("", max_length=60), limit: int = Query(20, ge=1, le=50),
           atlas=Depends(get_atlas)):
    return search_fighters(atlas, q, limit=limit)


@app.get("/api/fighters/{fighter_id}", response_model=FighterProfile,
         responses={**NOT_FOUND, **UNAVAILABLE_RESPONSE})
def fighter(fighter_id: str = Path(..., pattern=FIGHTER_ID_PATTERN),
            atlas=Depends(get_atlas)):
    return service.fighter_profile(atlas, fighter_id)


@app.get("/api/fighters/{fighter_id}/similar", response_model=SimilarResponse,
         responses={**NOT_FOUND, **UNAVAILABLE_RESPONSE})
def similar(
    fighter_id: str = Path(..., pattern=FIGHTER_ID_PATTERN),
    n: int = Query(10, ge=1, le=100),
    same_cluster: bool = False,
    division: Optional[str] = Query(None, max_length=60,
                                    description="'same' or a division name"),
    category: Optional[Literal["same", "men", "women"]] = None,
    min_minutes: Optional[float] = Query(None, ge=0),
    atlas=Depends(get_atlas),
):
    try:
        return find_similar(atlas, fighter_id, n=n, same_cluster=same_cluster,
                            division=division, category=category, min_minutes=min_minutes)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/fighters/{fighter_id}/fights", response_model=FightsResponse,
         responses={**NOT_FOUND, **UNAVAILABLE_RESPONSE})
def fights(fighter_id: str = Path(..., pattern=FIGHTER_ID_PATTERN),
           limit: int = Query(60, ge=1, le=200), bouts=Depends(get_bouts)):
    return service.fighter_fights(bouts, fighter_id, limit=limit)


@app.get("/api/recent", response_model=RecentResponse, responses=UNAVAILABLE_RESPONSE)
def recent(request: Request, atlas=Depends(get_atlas), bouts=Depends(get_bouts)):
    return _cached(request, atlas, "recent", lambda a: service.recent_activity(a, bouts))


@app.get("/api/map", response_model=MapResponse, responses=UNAVAILABLE_RESPONSE)
def style_map(request: Request, response: Response, atlas=Depends(get_atlas)):
    response.headers["Cache-Control"] = "public, max-age=300"
    return _cached(request, atlas, "map", service.style_map)


@app.get("/api/compare", response_model=CompareResponse,
         responses={**NOT_FOUND, **UNAVAILABLE_RESPONSE})
def compare(ids: list[str] = Query(default=[]), atlas=Depends(get_atlas)):
    _check_ids(ids)
    return service.compare(atlas, ids)


@app.get("/api/compare.csv", responses={**NOT_FOUND, **UNAVAILABLE_RESPONSE})
def compare_csv(ids: list[str] = Query(default=[]), atlas=Depends(get_atlas)):
    _check_ids(ids)
    text = service.compare_csv(service.compare(atlas, ids))
    return Response(
        content=text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="cage-atlas-comparison.csv"'},
    )
