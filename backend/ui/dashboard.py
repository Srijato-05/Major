"""
Visual Telemetry & Latency Dashboard
Tracks operational throughput (tons/hr), class yield counters, and latency jitter histograms.
"""

class TelemetryDashboard:
    def __init__(self):
        self.class_counts = {i: 0 for i in range(6)}
        self.total_ejections = 0
        self.missed_ejections = 0

    def record_ejection(self, class_id: int, success: bool = True):
        if success:
            self.class_counts[class_id] += 1
            self.total_ejections += 1
        else:
            self.missed_ejections += 1

    def get_summary(self) -> dict:
        return {
            "class_counts": self.class_counts,
            "total_ejections": self.total_ejections,
            "missed_ejections": self.missed_ejections
        }
