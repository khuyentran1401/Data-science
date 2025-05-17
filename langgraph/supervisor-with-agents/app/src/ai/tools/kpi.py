from typing import List
from src.ai.models.kpi import KPI
from src.repositories.kpi import FakeKPIRepository

repo = FakeKPIRepository()

def list_available_kpis() -> str:
    """
    Lists all available KPI metric names.
    
    This function retrieves all KPIs from the repository and returns their metric names as a comma-separated string.
    
    Returns:
        str: A comma-separated string of all available KPI metric names.
    """
    kpis = fetch_kpis()
    return ", ".join([k.metric for k in kpis])

def fetch_kpis() -> List[KPI]:
    """
    Retrieve all Key Performance Indicators (KPIs) from the repository.

    This function accesses the KPI repository to fetch a list of all available KPIs.

    Returns:
        List[KPI]: A list containing all KPI objects retrieved from the repository.

    Raises:
        RepositoryError: If there is an issue accessing the KPI repository.
    """
    return repo.fetch_all()

def get_kpi_by_metric(metric_name: str) -> KPI:
    """
    Retrieve a specific KPI by its metric name.

    This function searches the KPI repository for a KPI that matches the provided metric name.

    Args:
        metric_name (str): The name of the metric to search for.

    Returns:
        KPI: The KPI object corresponding to the specified metric name.

    Raises:
        ValueError: If no KPI with the specified metric name is found.
        RepositoryError: If there is an issue accessing the KPI repository.
    """
    try:
        kpi = repo.get_by_metric(metric_name)
        return (
            f"KPI: {kpi.metric}\n"
            f"Value: {kpi.value} {kpi.unit}\n"
            f"Target: {kpi.target} {kpi.unit}\n"
            f"Trend: {kpi.trend}\n"
            f"Last updated: {kpi.last_updated}"
        )
    except ValueError as e:
        return f"No KPI found with the name '{metric_name}'. Please try one of the known metrics like 'Revenue' or 'Churn Rate'."
    except Exception as e:
        return f"An error occurred while fetching the KPI: {str(e)}"

