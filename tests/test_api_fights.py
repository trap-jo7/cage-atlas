import pytest
from fastapi.testclient import TestClient

import backend.main as main
from analysis.cageatlas.service import prepare_bouts
from tests.test_fights import make_bouts


@pytest.fixture()
def client():
    index = prepare_bouts(make_bouts())
    main.app.dependency_overrides[main.get_bouts] = lambda: index
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_fights_endpoint_shape(client):
    r = client.get("/api/fighters/a/fights")
    assert r.status_code == 200
    body = r.json()
    assert body["total_fights"] == 3 and body["note"]
    assert body["fights"][0]["date"] == "2024-01-01" and body["fights"][0]["how_won"]
    assert body["fights"][2]["fighter"]["sig_landed"] is None


def test_fights_errors(client):
    assert client.get("/api/fighters/nobody/fights").status_code == 404
    assert client.get("/api/fighters/bad%21id/fights").status_code == 422
    assert client.get("/api/fighters/a/fights", params={"limit": 0}).status_code == 422
    assert len(client.get("/api/fighters/a/fights", params={"limit": 1}).json()["fights"]) == 1


def test_fights_unavailable_gives_503_without_leaking_paths(monkeypatch):
    main.app.dependency_overrides.clear()
    main.app.state.bouts = None

    def boom():
        raise FileNotFoundError("/private/place/bouts.parquet")

    monkeypatch.setattr(main.service, "load_bouts", boom)
    try:
        r = TestClient(main.app).get("/api/fighters/a/fights")
        assert r.status_code == 503 and "/private/place" not in r.text
    finally:
        main.app.state.bouts = None


def test_recent_endpoint(client):
    from tests.test_service import atlas_with_map
    atlas = atlas_with_map()
    main.app.dependency_overrides[main.get_atlas] = lambda: atlas
    r = client.get("/api/recent")
    assert r.status_code == 200
    body = r.json()
    assert body["latest_event_date"] and "results" in body and "fighters" in body
