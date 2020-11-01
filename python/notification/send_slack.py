from knockknock import slack_sender

webhook_url = "webhook_url_to_your_slack_room"

@slack_sender(webhook_url=webhook_url, channel="your_favorite_slack_channel", user_mentions=["your_slack_id", "your_teammate_slack_id"])
def main():
    even_arr = []
    for i in range(10000):
        if i%2==0:
            even_arr.append(i)

if __name__=='__main__':
    main()
