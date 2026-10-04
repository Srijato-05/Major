import pytest
import numpy as np
from src.sorting.advanced_decision_engine import AdvancedSortingDecisionEngine
from src.telemetry.advanced_telemetry_engine import AdvancedTelemetryEngine
from src.simulation.digital_twin import ConveyorDigitalTwinSimulator

def test_advanced_decision_matrix():
    engine = AdvancedSortingDecisionEngine()

    # 1. High-Grade PCB exceeding 10 cm^2 (EU Annex VII mandatory extraction)
    res_large_pcb = engine.evaluate_item(track_id=1, class_id=0, confidence=0.92, area_mm2=1200.0) # 12 cm^2
    assert res_large_pcb.is_annex_vii_mandatory is True
    assert res_large_pcb.destination_action == "MANDATORY_ANNEX_VII_EXTRACTION"
    assert "Au (Gold)" in res_large_pcb.critical_materials_recovered
    assert res_large_pcb.commodity_value_eur > 0

    # 2. Battery pack with low confidence (triggering CPCB EPR Regime 2 safety penalty)
    res_battery_low_conf = engine.evaluate_item(track_id=2, class_id=2, confidence=0.45, area_mm2=800.0)
    assert res_battery_low_conf.is_hazardous is True
    assert res_battery_low_conf.penalty_eur > 0
    assert "PENALTY" in res_battery_low_conf.epr_compliance_status

def test_advanced_telemetry_accounting():
    telem = AdvancedTelemetryEngine(window_seconds=5.0, power_kw=4.5, electricity_cost_kwh=0.22)
    engine = AdvancedSortingDecisionEngine()

    item = engine.evaluate_item(track_id=1, class_id=1, confidence=0.95, area_mm2=500.0)
    telem.record_item(item)
    telem.record_latency(15.0)

    m = telem.get_dashboard_metrics()
    assert m["total_items_processed"] == 1
    assert m["gross_revenue_eur"] > 0
    assert m["energy_consumed_kwh"] >= 0
    assert "instant_throughput_tons_hr" in m
    assert m["fps"] > 0

def test_digital_twin_simulator_single_frame():
    sim = ConveyorDigitalTwinSimulator()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    res = sim.process_frame(rgb)

    assert "annotated_frame" in res
    assert "telemetry" in res
    assert "latency_ms" in res
    assert res["annotated_frame"].shape == (480, 640, 3)
