from prefect import flow, task


@task
def add_one(x):
    if x == 2:
        raise Exception("Raised exception")
    return x + 1


@task
def add_two(x):
    if x == 2:
        raise Exception("Raised exception")
    return x + 1


@flow
def main(nums=[1, 2, 3]):

    res = add_one.map(nums)
    return add_two.map(res)


if __name__ == "__main__":
    main()
