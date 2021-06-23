from eventsourcing.domain import Aggregate, event
from loguru import logger

class Person(Aggregate):
    @event('Created')
    def __init__(self):
        self.life = []

    @event("LifeHappened")
    def deal_with_life(self, some_event):
        self.life.append(some_event)

    @event("Grow")
    def learn_from_life(self, some_event):
        self.life.append(some_event)

@logger.catch
def main():
    ben = Person()
    assert ben.life == []

    ben.deal_with_life('school')
    ben.deal_with_life('friend')
    ben.learn_from_life('leave_home')

    assert ben.life == ['school', 'friend', 'leave_home']
    events  = ben.collect_events()
    assert len(events) == 4
    assert type(events[0]).__qualname__ == 'Person.Created'
    assert type(events[1]).__qualname__ == 'Person.LifeHappened'
    assert type(events[2]).__qualname__ == 'Person.LifeHappened'
    assert type(events[3]).__qualname__ == 'Person.Grow'

    

main()
