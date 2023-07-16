from loguru import logger


def division(a, b):
    return a / b


def nested(c):
    try:
        division(1, c)
    except ZeroDivisionError:
        logger.exception("ZeroDivisionError")


if __name__ == "__main__":
    nested(0)
