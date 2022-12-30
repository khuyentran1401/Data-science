from prefect import flow, task


@task
def add_one(x):
    print(f"add_one({x})")
    if x == 2:
        raise Exception("Raised exception")
    return x + 1


@flow
def main(nums=[1, 2, 3]):

    return add_one.map(nums)


if __name__ == "__main__":
    main()
