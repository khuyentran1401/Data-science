from loguru import logger


def main():
    logger.debug("Loaded 1000 rows from dataset.csv")
    logger.info("Started training RandomForest model")
    logger.warning("Missing values detected in 'age' column")
    logger.error("Model training failed: insufficient memory")


if __name__ == "__main__":
    main()
