def encode_data(data: list):
    data_map = {"a": 1, "b": 2, "c": 3}
    return [data_map[num] for num in data]


def add_one(data: list):
    return [num + 1 for num in data]


def add_two(data: list):
    return [num + 2 for num in data]


def multiply_by_two(data: list):
    return [num * 2 for num in data]


def process_data(data: list):
    data = encode_data(data)
    print(f"Encoded data: {data}")
    data = add_one(data)
    print(f"Added one: {data}")
    data = add_two(data)
    print(f"Added two: {data}")
    data = multiply_by_two(data)
    print(f"Multiplied by two: {data}")
    return data


process_data(["a", "a", "c"])
