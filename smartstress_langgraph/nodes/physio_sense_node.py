"""PhysioSense node backed by the frozen WESAD Attention-DNN."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict

from ..physio import (
    MODEL_ID,
    WesadAttentionPredictor,
    prepare_physio_features,
)
from ..state import SmartStressState, append_audit_event, append_error


@lru_cache(maxsize=1)
def _get_predictor() -> WesadAttentionPredictor:
    """Load and checksum-validate the checkpoint once per process."""
    return WesadAttentionPredictor()


def _run_stress_model(raw_sensor_input: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare one sensor payload and return paper-model inference fields."""
    model_input = dict(raw_sensor_input)
    model_input.pop("timestamp", None)
    prepared = prepare_physio_features(**model_input)
    predictor = _get_predictor()
    prediction = predictor.predict(prepared.values)
    return {
        "current_stress_prob": prediction.probability,
        "stress_detected": prediction.is_stress,
        "stress_threshold": prediction.threshold,
        "physio_model_id": prediction.model_id,
        "physio_input_source": prepared.source,
        "physio_features": list(prepared.values),
        "physio_feature_map": prepared.as_dict(),
        "physio_raw_features": list(prepared.raw_values or ()),
        "physio_baseline_features": list(prepared.baseline_values or ()),
    }


def physio_sense_node(state: SmartStressState) -> Dict[str, Any]:
    """Infer stress only when a new validated sensor payload is present."""
    raw_data = state.get("raw_sensor_input")
    if not raw_data:
        return {}

    try:
        inference = _run_stress_model(raw_data)
    except Exception as exc:  # Convert model/input failures into observable state.
        append_error(state, f"PhysioSense inference failure: {exc}")
        append_audit_event(
            state,
            node_name="physio_sense",
            summary="Physiological inference failed",
            details={"error": str(exc)},
        )
        return {
            "raw_sensor_input": None,
            "error_log": list(state.get("error_log", [])),
            "audit_trail": list(state.get("audit_trail", [])),
        }

    sensor_timestamp = raw_data.get("timestamp")
    if not sensor_timestamp:
        sensor_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    history = [*state.get("stress_history", []), inference["current_stress_prob"]]
    timestamps = [*state.get("stress_timestamps", []), str(sensor_timestamp)]

    append_audit_event(
        state,
        node_name="physio_sense",
        summary="Inferred stress probability with frozen Attention-DNN",
        details={
            "model_id": MODEL_ID,
            "input_source": inference["physio_input_source"],
            "current_stress_prob": inference["current_stress_prob"],
            "stress_detected": inference["stress_detected"],
            "threshold": inference["stress_threshold"],
        },
    )
    return {
        **inference,
        "stress_history": history,
        "stress_timestamps": timestamps,
        "physio_attributions": {},
        "physio_top_drivers": [],
        "raw_sensor_input": None,
        "audit_trail": list(state.get("audit_trail", [])),
    }
