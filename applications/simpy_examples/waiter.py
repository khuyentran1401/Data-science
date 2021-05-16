import simpy 

def waiter(env):
    while True: # Simulate until the time limit
        print(f"Start taking orders from customers at {env.now}")
        take_order_duration = 5
        yield env.timeout(take_order_duration) # models duration

        print(f'Start giving the orders to the cooks at {env.now}')
        give_order_duration = 2
        yield env.timeout(give_order_duration)

        print(f'Start serving customers food at {env.now}\n')
        serve_order_duration = 5
        yield env.timeout(serve_order_duration)

env = simpy.Environment() # the environment where the waiter lives
env.process(waiter(env)) # pass the waiter to the environment
env.run(until=30) # Run simulation until 30s