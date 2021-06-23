from eventsourcing.domain import Aggregate, event
from loguru import logger

class Person(Aggregate):
    @event('Created')
    def __init__(self):
        self.life = []

    @event("LifeHappened")
    def deal_with_life(self, some_event):
        self.life.append(some_event)

@logger.catch
def main():
    ben = Person()
    assert ben.life == []

    ben.deal_with_life('school')
    assert ben.life == ['school']
    assert len(ben.collect_events()) == 2

main()
