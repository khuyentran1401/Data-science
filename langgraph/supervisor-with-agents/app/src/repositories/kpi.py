from typing import List
from datetime import datetime, timedelta
from src.ai.models.kpi import KPI

class FakeKPIRepository:
    def fetch_all(self) -> List[KPI]:
        now = datetime.now()
        return [
            KPI(
                metric="Revenue",
                value=123456.78,
                unit="USD",
                target=150000.00,
                trend="up",
                last_updated=now - timedelta(days=1),
                critical_threshold=100000.00,
                status="on_track",
                owner="Finance Team"
            ),
            KPI(
                metric="New Users",
                value=321,
                unit="users",
                target=500,
                trend="down",
                last_updated=now - timedelta(hours=6),
                critical_threshold=250,
                status="at_risk",
                owner="Growth Team"
            ),
            KPI(
                metric="Churn Rate",
                value=4.2,
                unit="%",
                target=3.0,
                trend="stable",
                last_updated=now,
                critical_threshold=5.0,
                status="off_track",
                owner="Customer Success"
            )
        ]

    def get_by_metric(self, metric_name: str) -> KPI:
        return next(
            (kpi for kpi in self.fetch_all() if kpi.metric.lower() == metric_name.lower()),
            ValueError(f"KPI '{metric_name}' not found.")
        )
