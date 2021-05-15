import simpy 
from random import random, seed


def customer(env, name, restaurant, **duration):
    while True:
        yield env.timeout(random()*10) # There is a new customer between 0 and 10 minutes
        print(f"{name} enters the restaurant and for the waiter to come at {round(env.now, 2)}")
        with restaurant.request() as req:
            yield req 

            print(f"Sits are available. {name} get sitted at {round(env.now, 2)}")
            yield env.timeout(duration['get_sitted'])

            print(f"{name} starts looking at the menu at {round(env.now, 2)}")
            yield env.timeout(duration['choose_food'])

            print(f'Waiters start getting the order from {name} at {round(env.now, 2)}')
            yield env.timeout(duration['give_order'])

            print(f'{name} starts waiting for food at {round(env.now, 2)}')
            yield env.timeout(duration['wait_for_food'])

            print(f'{name} starts eating at {round(env.now, 2)}')
            yield env.timeout(duration['eat'])

            print(f'{name} starts paying at {round(env.now, 2)}')
            yield env.timeout(duration['pay'])

            print(f'{name} leaves at {round(env.now, 2)}')


seed(1)
env = simpy.Environment()
restaurant = simpy.Resource(env, capacity=2)
durations = {'get_sitted': 1, 'choose_food': 10, 'give_order': 5, 'wait_for_food': 20, 'eat': 45, 'pay': 10}

for i in range(5):
    env.process(customer(env, f'Customer {i}', restaurant, **durations))

env.run(until=95)