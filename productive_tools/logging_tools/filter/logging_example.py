import logging

logging.basicConfig(
    filename="hello.log",
    format="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)


class CustomFilter(logging.Filter):
    def filter(self, record):
        return "Hello" in record.msg


# Create a custom logging filter
custom_filter = CustomFilter()

# Get the root logger and add the custom filter to it
logger = logging.getLogger()
logger.addFilter(custom_filter)


def main():
    logger.info("Hello World")
    logger.info("Bye World")


if __name__ == "__main__":
    main()
