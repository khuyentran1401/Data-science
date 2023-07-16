from loguru import logger

logger.add("hello.log", filter=lambda x: "Hello" in x["message"], level="INFO")


def main():
    logger.info("Hello World")
    logger.info("Bye World")


if __name__ == "__main__":
    main()
