import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logging.debug("Loaded 1000 rows from dataset.csv")
    logging.info("Started training RandomForest model")
    logging.warning("Missing values detected in 'age' column")
    logging.error("Model training failed: insufficient memory")


if __name__ == "__main__":
    main()
