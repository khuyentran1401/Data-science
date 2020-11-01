from knockknock import email_sender


@email_sender(recipient_emails=["khuyentran1476@gmail.com"], sender_email="khuyentran7676@gmail.com")
def email():
    even_arr = []
    for i in range(10000):
        if i%2==0:
            even_arr.append(i)




if __name__=='__main__':
    main()

