from typing import List
from datetime import date, datetime
from src.ai.models.customer import CustomerProfile


class FakeCustomerRepository:
    def get_profile(self, customer_id: str) -> CustomerProfile:
        return CustomerProfile(
            id=customer_id,
            name="Jane Doe",
            score=87,
            last_purchase="2025-04-20",
            email="jane.doe@example.com",
            phone="+1-555-1234",
            address="123 Main St, Anytown, USA",
            purchase_history=[
                {"date": "2025-04-20", "item": "Widget A", "amount": 29.99},
                {"date": "2025-03-15", "item": "Widget B", "amount": 49.99},
                {"date": "2025-02-10", "item": "Widget C", "amount": 15.00},
            ],
            preferences={
                "contact_method": "email",
                "interests": ["gadgets", "home automation"],
                "marketing_opt_in": True
            },
            loyalty={
                "program_member": True,
                "points": 2400,
                "tier": "Gold"
            }
        )

    def get_feedback(self, customer_id: str) -> List[dict]:
        return [
            {
                "date": "2025-04-21",
                "feedback": "Great service!",
                "channel": "email",
                "rating": 5
            },
            {
                "date": "2025-03-16",
                "feedback": "Product quality could be better.",
                "channel": "web",
                "rating": 3
            }
        ]

    def get_support_tickets(self, customer_id: str) -> List[dict]:
        return [
            {
                "ticket_id": "T123",
                "issue": "Late delivery",
                "status": "Resolved",
                "opened": "2025-04-19",
                "resolved": "2025-04-20"
            },
            {
                "ticket_id": "T124",
                "issue": "Wrong item received",
                "status": "Open",
                "opened": "2025-04-22",
                "resolved": None
            }
        ]

    def is_vip(self, customer_id: str) -> bool:
        # Simula uma lógica baseada em pontos de fidelidade
        profile = self.get_profile(customer_id)
        return profile.loyalty["points"] > 2000

    def get_last_interaction_date(self, customer_id: str) -> str:
        # Usa a última data entre compra, feedback e suporte
        dates = [datetime.strptime("2025-04-20", "%Y-%m-%d")]
        for feedback in self.get_feedback(customer_id):
            dates.append(datetime.strptime(feedback["date"], "%Y-%m-%d"))
        for ticket in self.get_support_tickets(customer_id):
            dates.append(datetime.strptime(ticket["opened"], "%Y-%m-%d"))
        return max(dates).strftime("%Y-%m-%d")
