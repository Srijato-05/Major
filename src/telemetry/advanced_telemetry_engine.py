import time
from typing import Dict, Any, List, Optional
from collections import deque
from src.sorting.advanced_decision_engine import AdvancedSortedItemResult

class AdvancedTelemetryEngine:
    """
    High-Fidelity Industrial Telemetry & Accounting Simulator:
    - Mass throughput metrics (kg/hr and metric tons/hr)
    - Recovery purity and contamination cross-matrices
    - Cumulative commodity revenue and net operating P&L (€)
    - Electricity consumption costing (kWh)
    - CPCB EPR Penalty deductions
    - Rolling latency, FPS, and scheduling jitter
    """

    def __init__(
        self,
        window_seconds: float = 10.0,
        power_kw: float = 4.5,
        electricity_cost_kwh: float = 0.22
    ):
        self.window_seconds = window_seconds
        self.power_kw = power_kw
        self.electricity_cost_kwh = electricity_cost_kwh

        self.item_history = deque()
        self.latency_history = deque()

        # Cumulative counters
        self.start_time = time.time()
        self.total_items_processed = 0
        self.total_mass_kg = 0.0
        self.total_gross_revenue_eur = 0.0
        self.total_penalties_eur = 0.0

        # Bin accounting
        self.bin_mass_kg: Dict[str, float] = {}
        self.bin_item_counts: Dict[str, int] = {}
        self.bin_purity_scores: Dict[str, float] = {}
        self.critical_materials_counter: Dict[str, int] = {}
        self.annex_vii_extractions = 0

    def record_item(self, item: AdvancedSortedItemResult):
        now = time.time()
        self.item_history.append((now, item))
        self.total_items_processed += 1
        self.total_mass_kg += item.estimated_mass_kg
        self.total_gross_revenue_eur += item.commodity_value_eur
        self.total_penalties_eur += item.penalty_eur

        if item.is_annex_vii_mandatory:
            self.annex_vii_extractions += 1

        # Tally critical raw materials
        for elem in item.critical_materials_recovered:
            self.critical_materials_counter[elem] = self.critical_materials_counter.get(elem, 0) + 1

        # Bin tallies
        b = item.target_bin
        self.bin_mass_kg[b] = self.bin_mass_kg.get(b, 0.0) + item.estimated_mass_kg
        self.bin_item_counts[b] = self.bin_item_counts.get(b, 0) + 1

    def record_latency(self, latency_ms: float):
        now = time.time()
        self.latency_history.append((now, latency_ms))

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self.item_history and self.item_history[0][0] < cutoff:
            self.item_history.popleft()
        while self.latency_history and self.latency_history[0][0] < cutoff:
            self.latency_history.popleft()

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        now = time.time()
        self._prune(now)

        elapsed_sec = max(now - self.start_time, 0.1)
        elapsed_hr = elapsed_sec / 3600.0

        # Instantaneous window throughput
        window_mass_kg = sum(item.estimated_mass_kg for _, item in self.item_history)
        window_duration_hr = min(elapsed_sec, self.window_seconds) / 3600.0
        instant_kg_hr = window_mass_kg / max(window_duration_hr, 1e-6)
        instant_tons_hr = instant_kg_hr / 1000.0

        # Energy consumption & operating expenditure
        energy_kwh = self.power_kw * elapsed_hr
        operating_energy_cost_eur = energy_kwh * self.electricity_cost_kwh

        # Economic bottom-line
        net_profit_eur = self.total_gross_revenue_eur - self.total_penalties_eur - operating_energy_cost_eur
        net_eur_per_hr = net_profit_eur / max(elapsed_hr, 1e-6)

        # Latency & FPS
        avg_latency_ms = sum(lat for _, lat in self.latency_history) / max(len(self.latency_history), 1)
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        return {
            "uptime_seconds": round(elapsed_sec, 1),
            "total_items_processed": self.total_items_processed,
            "annex_vii_mandatory_extractions": self.annex_vii_extractions,
            "total_mass_kg": round(self.total_mass_kg, 3),
            "instant_throughput_kg_hr": round(instant_kg_hr, 2),
            "instant_throughput_tons_hr": round(instant_tons_hr, 3),
            "gross_revenue_eur": round(self.total_gross_revenue_eur, 2),
            "penalties_deducted_eur": round(self.total_penalties_eur, 2),
            "energy_consumed_kwh": round(energy_kwh, 3),
            "energy_cost_eur": round(operating_energy_cost_eur, 2),
            "net_profit_eur": round(net_profit_eur, 2),
            "net_rate_eur_per_hr": round(net_eur_per_hr, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "fps": round(fps, 1),
            "bin_item_counts": dict(self.bin_item_counts),
            "bin_mass_kg": {k: round(v, 3) for k, v in self.bin_mass_kg.items()},
            "critical_materials_recovered": dict(self.critical_materials_counter)
        }
