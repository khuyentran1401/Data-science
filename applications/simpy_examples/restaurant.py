import simpy 
import random 
from collections import namedtuple

RANDOM_SEED = 20
NUM_ITEMS = 10 # Number of items per food option
SIM_TIME  = 240

def customer(env, food, num_food_order, restaurant):
    """Customer tries to order a certain number of a particular food, 
    if that food ran out, customer leaves. If there is enough food left,
    customer orders food."""

    with restaurant.staff.request() as customer:

        # If there is not enough food left, customer leaves
        if restaurant.available[food] < num_food_order:
            restaurant.rejected_customers[food] +=1
            return

        # If there is enough food left, customer orders food
        restaurant.available[food] -= num_food_order
        # The time it takes to prepare food
        yield env.timeout(10*num_food_order)

        # If there is no food left after customer orders, trigger run out event
        if restaurant.available[food] == 0:
            restaurant.run_out[food].succeed()
            restaurant.when_run_out[food] = env.now

        yield env.timeout(2)


def customer_arrivals(env, restaurant):
    """Create new customers until the simulation reaches the time limit"""
    while True:
        yield env.timeout(random.random()*10)

        food = random.choice(restaurant.foods)
        num_food_order = random.randint(1,6)

        if restaurant.available[food]:
            env.process(customer(env, food, num_food_order, restaurant))

# Set up and start the simulation
random.seed(RANDOM_SEED)
env = simpy.Environment()

# Create restaurant
staff = simpy.Resource(env, capacity=1)
foods = ['Spicy Chicken', 'Poached Chicken', 'Tomato Chicken Skillet', 'Honey Mustard Chicken']
available = {food: NUM_ITEMS for food in foods}
run_out = {food: env.event() for food in foods}
when_run_out = {food: None for food in foods}
rejected_customers = {food: 0 for food in foods}

Restaurant = namedtuple('Restaurant', 'staff, foods, available,'
                        'run_out, when_run_out, rejected_customers')
restaurant = Restaurant(staff, foods, available, run_out,
                        when_run_out, rejected_customers)

# Start process and run
env.process(customer_arrivals(env, restaurant))
env.run(until=SIM_TIME)

for food in foods:
    if restaurant.run_out[food]:
        print(f'The {food} ran out {round(restaurant.when_run_out[food], 2)} '
            'minutes after the restaurant opens.')
        print(f'Number of people leaving queue when the {food} ran out is ' 
        f'{restaurant.rejected_customers[food]}.\n')