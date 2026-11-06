import csv
import io

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from analysis.cageatlas.similarity import find_similar
from tests.test_service import atlas_with_map


@pytest.fixture()
def atlas():
    return atlas_with_map()


@pytest.fixture()
def client(atlas):
    main.app.dependency_overrides[main.get_atlas] = lambda: atlas
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_status_labels_data_mode(client):
    body = client.get("/api/status").json()
    assert body["data_mode"] == "real_snapshot"
    assert body["organizations"][0]["data_through"] == "2026-05-16"
    assert body["attribution"] and body["limits"]


def test_search(client):
    r = client.get("/api/fighters/search", params={"q": "fighter 01"})
    assert r.status_code == 200 and r.json()
    assert all("fighter 01" in h["fighter_name"].lower() for h in r.json())
    assert client.get("/api/fighters/search", params={"q": "x"}).json() == []
    assert client.get("/api/fighters/search", params={"q": "fi", "limit": 0}).status_code == 422
    assert client.get("/api/fighters/search", params={"q": "a" * 61}).status_code == 422


def test_profile_found_missing_and_bad_id(client):
    r = client.get("/api/fighters/f000")
    assert r.status_code == 200
    body = r.json()
    td = next(f for f in body["features"] if f["feature"] == "td_accuracy")
    assert td["undefined"] is True and td["value"] is None
    assert client.get("/api/fighters/nope").status_code == 404
    assert client.get("/api/fighters/bad%21id").status_code == 422


def test_similar_matches_the_shared_package(client, atlas):
    r = client.get("/api/fighters/f010/similar",
                   params={"n": 15, "division": "same", "category": "same", "min_minutes": 30})
    assert r.status_code == 200
    direct = find_similar(atlas, "f010", n=15, division="same", category="same", min_minutes=30.0)
    got, want = r.json()["results"], direct["results"]
    assert [x["fighter_id"] for x in got] == [x["fighter_id"] for x in want]
    assert [x["distance"] for x in got] == [x["distance"] for x in want]
    assert "f010" not in [x["fighter_id"] for x in got]


def test_similar_empty_set_is_a_normal_response_with_reason(client):
    r = client.get("/api/fighters/f010/similar", params={"division": "Strawweight"})
    assert r.status_code == 200
    assert r.json()["results"] == [] and r.json()["empty_reason"]


def test_similar_rejects_bad_inputs(client):
    base = "/api/fighters/f010/similar"
    assert client.get(base, params={"n": 0}).status_code == 422
    assert client.get(base, params={"n": 101}).status_code == 422
    assert client.get(base, params={"n": "ten"}).status_code == 422
    assert client.get(base, params={"category": "everyone"}).status_code == 422
    assert client.get(base, params={"min_minutes": -1}).status_code == 422
    assert client.get("/api/fighters/nope/similar").status_code == 404


def test_map_is_cached_and_complete(client, atlas):
    r = client.get("/api/map")
    assert r.status_code == 200
    assert len(r.json()["points"]) == len(atlas.artifact["cohort_ids"])
    assert "max-age" in r.headers["cache-control"]
    assert r.json()["axes"]["explained_variance_total"] == 0.5


def test_divisions(client):
    body = client.get("/api/divisions").json()
    assert body["divisions"] and all(d["fighters"] > 0 for d in body["divisions"])


def test_compare_and_validation(client):
    ok = client.get("/api/compare", params={"ids": ["f001", "f002", "f000"]})
    assert ok.status_code == 200
    assert len(ok.json()["metrics"]) == 9
    assert client.get("/api/compare", params={"ids": ["f001"]}).status_code == 422
    assert client.get("/api/compare", params={"ids": ["f001", "f001"]}).status_code == 422
    five = ["f001", "f002", "f003", "f004", "f005"]
    assert client.get("/api/compare", params={"ids": five}).status_code == 422
    assert client.get("/api/compare", params={"ids": ["f001", "bad!id"]}).status_code == 422
    assert client.get("/api/compare", params={"ids": ["f001", "missing"]}).status_code == 404


def test_compare_csv_download(client):
    r = client.get("/api/compare.csv", params={"ids": ["f001", "f002"]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0][:3] == ["metric", "unit", "role"] and len(rows[0]) == 5
    assert client.get("/api/compare.csv", params={"ids": ["f001"]}).status_code == 422


def test_unavailable_model_gives_503_without_leaking_paths_then_recovers(monkeypatch, atlas):
    main.app.dependency_overrides.clear()
    main.app.state.atlas = None
    calls = {"n": 0}

    def fake_load():
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("/private/place/model.joblib")
        return atlas

    monkeypatch.setattr(main, "load_atlas", fake_load)
    c = TestClient(main.app)
    try:
        bad = c.get("/api/status")
        assert bad.status_code == 503
        assert "could not be loaded" in bad.json()["detail"]
        assert "/private/place" not in bad.text
        assert c.get("/api/status").status_code == 200       # retry succeeds
        assert c.get("/api/health").status_code == 200
    finally:
        main.app.state.atlas = None
