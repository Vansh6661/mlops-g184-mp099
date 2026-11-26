from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    payload = {
        "rainfall_mm": 100.0,
        "fertilizer_kg": 50.0,
        "sowing_delay_days": 2,
        "irrigation_count": 3,
        "soil_ph": 6.5,
        "farm_acres": 5.0,
        "seed_variety": "HD-2967",
        "tehsil": "Bhopal",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))


def test_farm_acres_above_25_rejected():
    payload = {
        "rainfall_mm": 100.0,
        "fertilizer_kg": 50.0,
        "sowing_delay_days": 2,
        "irrigation_count": 3,
        "soil_ph": 6.5,
        "farm_acres": 26.0,
        "seed_variety": "HD-2967",
        "tehsil": "Bhopal",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_fertilizer_below_10_rejected():
    payload = {
        "rainfall_mm": 100.0,
        "fertilizer_kg": 5.0,
        "sowing_delay_days": 2,
        "irrigation_count": 3,
        "soil_ph": 6.5,
        "farm_acres": 5.0,
        "seed_variety": "HD-2967",
        "tehsil": "Bhopal",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422