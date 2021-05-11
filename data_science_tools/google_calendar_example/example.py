from calendar_api.calendar_api import google_calendar_api
import pickle 


with open('token.pickle', 'rb') as token:
    creds = pickle.load(token)
    
m=google_calendar_api()
print(creds)
# m.create_event(calendar_id='primary',
# start='2017,12,5,15,00,00',
# end='2017,12,5,15,15,00',
# description='foo'
# )