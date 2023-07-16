from loguru import logger


def division(a, b):
    return a / b


@logger.catch
def nested(c):
    division(1, c)


if __name__ == "__main__":
    nested(0)
