import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def division(a, b):
    return a / b


def nested(c):
    try:
        division(1, c)
    except ZeroDivisionError:
        logging.exception("ZeroDivisionError")


if __name__ == "__main__":
    nested(0)
