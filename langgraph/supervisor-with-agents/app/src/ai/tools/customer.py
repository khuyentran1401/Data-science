from typing import List
from src.ai.models.customer import CustomerProfile
from src.repositories.customer import FakeCustomerRepository

repo = FakeCustomerRepository()

def get_customer_profile(customer_id: str) -> CustomerProfile:
    """
    Fetches the customer profile based on the customer ID.
    Args:
        customer_id (str): The ID of the customer.
    Returns:
        CustomerProfile: The customer profile object.
    Raises:
        ValueError: If the customer ID is not found.
    """
    return repo.get_profile(customer_id)

def get_customer_feedback(customer_id: str) -> List[dict]:
    """
    Fetches the customer feedback based on the customer ID.
    Args:
        customer_id (str): The ID of the customer.
    Returns:
        List[dict]: A list of feedback dictionaries.
    Raises:
        ValueError: If the customer ID is not found.
    """
    return repo.get_feedback(customer_id)

def get_customer_support_tickets(customer_id: str) -> List[dict]:
    """
    Fetches the customer support tickets based on the customer ID.
    Args:
        customer_id (str): The ID of the customer.
    Returns:
        List[dict]: A list of support ticket dictionaries.
    Raises:
        ValueError: If the customer ID is not found.
    """    
    return repo.get_support_tickets(customer_id)

def search_customer_feedback(customer_id: str, keyword: str) -> List[dict]:
    """
    Searches customer feedback for a specific keyword.
    Args:
        customer_id (str): The ID of the customer.
        keyword (str): The keyword to search for in the feedback.
    Returns:
        List[dict]: A list of feedback dictionaries that contain the keyword.
    Raises:
        ValueError: If the customer ID is not found.
    """
    return [fb for fb in repo.get_feedback(customer_id) if keyword.lower() in fb["feedback"].lower()]

def search_support_tickets_by_status(customer_id: str, status: str) -> List[dict]:
    """
    Searches customer support tickets by their status.

    Args:
        customer_id (str): The ID of the customer.
        status (str): The status to filter tickets by (e.g., "open", "closed").
    Returns:
        List[dict]: A list of support ticket dictionaries that match the status.
    Raises:
        ValueError: If the customer ID is not found.
    """
    return [ticket for ticket in repo.get_support_tickets(customer_id) if ticket["status"].lower() == status.lower()]

def search_purchase_history_by_item(customer_id: str, item_name: str) -> List[dict]:
    """
    Search a customer's purchase history for items matching a given name.

    This function retrieves the customer's profile using their unique identifier
    and filters their purchase history to include only those purchases where the
    item name contains the specified keyword, case-insensitive.

    Args:
        customer_id (str): The unique identifier of the customer.
        item_name (str): The keyword to search for within the item names.

    Returns:
        List[dict]: A list of dictionaries representing purchases that match the
        search criteria. Each dictionary contains details of a purchase.

    Raises:
        ValueError: If the customer profile cannot be found.
    """
    profile = repo.get_profile(customer_id)
    return [purchase for purchase in profile.purchase_history if item_name.lower() in purchase["item"].lower()]
