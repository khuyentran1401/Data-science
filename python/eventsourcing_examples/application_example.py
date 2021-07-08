from eventsourcing.application import Application
from eventsourcing.domain import Aggregate, event
from eventsourcing.system import NotificationLogReader
from rich import print

class Person(Aggregate):
    @event('Created')
    def __init__(self):
        self.life = []

    @event("LifeHappened")
    def deal_with_life(self, some_event):
        self.life.append(some_event)

class Record(Application):
    def create_person(self):
        person = Person()
        self.save(person)
        return person.id 
    
    def deal_with_life(self, person_id, some_event):
        person = self.repository.get(person_id)
        person.deal_with_life(some_event)
        self.save(person)

    def view_life(self, person_id):
        person = self.repository.get(person_id)
        return person.life

def notification_to_table(record):
    return [record.id, record.version, record.created_on, record.modified_on, record.history]

def main():
    record = Record()
    ben = record.create_person()
    record.deal_with_life(ben, 'school')
    record.deal_with_life(ben, 'friend')
    assert record.view_life(ben) == ['school', 'friend']

    alex = record.create_person()
    assert alex != ben
    record.deal_with_life(alex, 'girl')
    assert record.view_life(alex) == ['girl']

    # Create event notification
    reader = NotificationLogReader(record.log)
    notifications = list(reader.read(start=1))
    assert len(notifications) == 5
    print(notifications)
    for notification in notifications:
        print(notification_to_table(notification))


main()
    