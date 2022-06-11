from prefect import flow, task
from pydantic import BaseModel


class Input(BaseModel):
    data: list
    attributes: list


@flow
def process_input(input: Input):
    print(type(input))
    print(input.data)
    ...


@flow
def main(data, attributes):
    input = {"data": data, "attributes": attributes}
    process_input(input)


if __name__ == "__main__":
    data = [{"a": [1, 2, 3], "b": [4, 5]}, {"a": [1, 2, 3], "b": [4, 5]}]
    attributes = ["a", "b"]

    main(data, attributes)
