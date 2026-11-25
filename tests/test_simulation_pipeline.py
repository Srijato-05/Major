import pytest
import numpy as np
from src.sorting.advanced_decision_engine import AdvancedSortingDecisionEngine
from src.telemetry.advanced_telemetry_engine import AdvancedTelemetryEngine
from src.simulation.digital_twin import ConveyorDigitalTwinSimulator

def test_advanced_decision_matrix():
    engine = AdvancedSortingDecisionEngine()

    # 1. High-Grade PCB exceeding 10 cm^2 (EU Annex VII mandatory extraction)
    res_large_pcb = engine.evaluate_item(track_id=1, class_id=1, confidence=0.92, area_mm2=1200.0) # 12 cm^2
    assert res_large_pcb.is_annex_vii_mandatory is True
    assert res_large_pcb.destination_action == "MANDATORY_ANNEX_VII_EXTRACTION"
    assert "Au (Gold)" in res_large_pcb.critical_materials_recovered
    assert res_large_pcb.commodity_value_eur > 0

    # 2. Battery pack with low confidence (triggering CPCB EPR Regime 2 safety penalty)
    res_battery_low_conf = engine.evaluate_item(track_id=2, class_id=3, confidence=0.45, area_mm2=800.0)
    assert res_battery_low_conf.is_hazardous is True
    assert res_battery_low_conf.penalty_eur > 0
    assert "PENALTY" in res_battery_low_conf.epr_compliance_status

def test_advanced_telemetry_accounting():
    telem = AdvancedTelemetryEngine(window_seconds=5.0, power_kw=4.5, electricity_cost_kwh=0.22)
    engine = AdvancedSortingDecisionEngine()

    item = engine.evaluate_item(track_id=1, class_id=2, confidence=0.95, area_mm2=500.0)
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
    assert "pneumatic_stats" in res
    assert "latency_ms" in res
    assert res["annotated_frame"].shape == (480, 640, 3)


def test_pneumatic_scheduler_and_bin_purity():
    from src.sorting.pneumatic_scheduler import PneumaticActuatorScheduler, ItemState

    sched = PneumaticActuatorScheduler(
        ejection_line_x_mm=1200.0,
        actuation_lag_ms=2.5,
        solenoid_pitch_mm=15.625,
        total_nozzles=64,
        speed_mps=2.5,
    )
    t0 = 1_000_000_000_000
    assert sched.update_and_schedule(1, 3, 200.0, 156.25, "BIN_BATTERY_ISOLATION", now_ns=t0) is None
    pulse = sched.update_and_schedule(1, 3, 200.0, 156.25, "BIN_BATTERY_ISOLATION", now_ns=t0 + 16_000_000)
    assert pulse is not None
    assert pulse.valve_idx == 10
    assert sched.items[1].state == ItemState.IN_TRANSIT

    due = sched.pop_due_pulses(now_ns=pulse.t_fire_ns)
    assert len(due) == 1
    assert sched.items[1].state == ItemState.EJECTED

    telem = AdvancedTelemetryEngine()
    telem.record_sorting_accuracy("BIN_BATTERY_ISOLATION", predicted_class_id=3, gt_class_id=3)
    telem.record_sorting_accuracy("BIN_BATTERY_ISOLATION", predicted_class_id=3, gt_class_id=5)
    m = telem.get_dashboard_metrics()
    assert m["bin_purity_scores"]["BIN_BATTERY_ISOLATION"] == 50.0
    assert m["class_recovery_rates"][3] == 100.0


def test_scrolling_conveyor_stream():
    from src.simulation.conveyor_stream import ScrollingConveyorStream

    stream = ScrollingConveyorStream(manifest_path="data/processed/test.json", speed_mps=2.5, fps=30)
    rgb1, spec1, gt1 = stream.step()
    rgb2, spec2, gt2 = stream.step()
    assert rgb1.shape == (640, 640, 3)
    assert spec1.shape == (640, 640)
    assert gt1.shape == (640, 640)
    assert not np.array_equal(rgb1, rgb2)
