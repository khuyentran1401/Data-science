import logging

# Create a logger and set the logging level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def encode_data(data: list):
    logger.info("Encode data")
    data_map = {"a": 1, "b": 2, "c": 3}
    logger.debug(f"Data map: {data_map}")
    return [data_map[num] for num in data]


def add_one(data: list):
    logger.info("Add one")
    return [num + 1 for num in data]


def process_data(data: list):
    logger.info("Process data")
    data = encode_data(data)
    logger.debug(f"Encoded data: {data}")
    data = add_one(data)
    logger.debug(f"Added one: {data}")


process_data(["a", "a", "c"])
