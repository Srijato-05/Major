import time
from typing import Dict, List, Any
from collections import deque
from src.sorting.decision_engine import SortedItemResult

class IndustrialTelemetryEngine:
    """
    Computes real-time industrial KPIs:
    - Mass throughput (kg/hr or tons/hr)
    - Recovery rate and purity per bin (%)
    - Economic yield (€/hr)
    - End-to-end pipeline latency (FPS / ms)
    """

    def __init__(self, window_seconds: float = 10.0, power_kw: float = 4.5, electricity_cost_kwh: float = 0.22):
        self.window_seconds = window_seconds
        self.power_kw = power_kw
        self.electricity_cost_kwh = electricity_cost_kwh

        # Time-windowed buffers for instantaneous rates
        self.item_history = deque()  # (timestamp, SortedItemResult)
        self.latency_history = deque()  # (timestamp, latency_ms)

        # Cumulative totals
        self.total_mass_kg = 0.0
        self.total_value_eur = 0.0
        self.total_items_count = 0
        self.bin_mass_totals: Dict[str, float] = {}
        self.bin_item_counts: Dict[str, int] = {}

        self.start_time = time.time()

    def record_item(self, item: SortedItemResult):
        now = time.time()
        self.item_history.append((now, item))
        self.total_mass_kg += item.estimated_mass_kg
        self.total_value_eur += item.commodity_value_eur
        self.total_items_count += 1

        self.bin_mass_totals[item.target_bin] = self.bin_mass_totals.get(item.target_bin, 0.0) + item.estimated_mass_kg
        self.bin_item_counts[item.target_bin] = self.bin_item_counts.get(item.target_bin, 0) + 1

    def record_latency(self, latency_ms: float):
        now = time.time()
        self.latency_history.append((now, latency_ms))

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self.item_history and self.item_history[0][0] < cutoff:
            self.item_history.popleft()
        while self.latency_history and self.latency_history[0][0] < cutoff:
            self.latency_history.popleft()

    def get_metrics(self) -> Dict[str, Any]:
        now = time.time()
        self._prune(now)

        elapsed_total_hr = max((now - self.start_time) / 3600.0, 1e-6)

        # Windowed throughput
        window_mass_kg = sum(item.estimated_mass_kg for _, item in self.item_history)
        window_duration_hr = min(now - self.start_time, self.window_seconds) / 3600.0
        window_duration_hr = max(window_duration_hr, 1e-6)

        throughput_kg_hr = window_mass_kg / window_duration_hr
        throughput_tons_hr = throughput_kg_hr / 1000.0

        # Economic yield
        operating_energy_cost_eur = self.power_kw * self.electricity_cost_kwh * elapsed_total_hr
        net_profit_eur = self.total_value_eur - operating_energy_cost_eur
        net_eur_per_hr = net_profit_eur / elapsed_total_hr

        # Latency & FPS
        avg_latency_ms = sum(lat for _, lat in self.latency_history) / max(len(self.latency_history), 1)
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        return {
            "uptime_seconds": round(now - self.start_time, 1),
            "total_items_sorted": self.total_items_count,
            "total_mass_kg": round(self.total_mass_kg, 3),
            "throughput_kg_hr": round(throughput_kg_hr, 2),
            "throughput_tons_hr": round(throughput_tons_hr, 3),
            "total_commodity_value_eur": round(self.total_value_eur, 2),
            "operating_cost_eur": round(operating_energy_cost_eur, 2),
            "net_profit_eur": round(net_profit_eur, 2),
            "net_rate_eur_per_hr": round(net_eur_per_hr, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "fps": round(fps, 1),
            "bin_breakdown": dict(self.bin_item_counts),
            "bin_mass_breakdown_kg": {k: round(v, 3) for k, v in self.bin_mass_totals.items()}
        }
