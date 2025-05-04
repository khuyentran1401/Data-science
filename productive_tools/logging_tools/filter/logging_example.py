import logging

logging.basicConfig(
    filename="hello.log",
    format="%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class CustomFilter(logging.Filter):
    def filter(self, record):
        return "Hello" in record.msg


# Get the root logger and add the custom filter to it
logger = logging.getLogger()
logger.addFilter(CustomFilter())


def main():
    logger.info("Hello World")
    logger.info("Bye World")


if __name__ == "__main__":
    main()
