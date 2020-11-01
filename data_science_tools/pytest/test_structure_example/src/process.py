from textblob import TextBlob

def extract_sentiment(text: str):
        '''Extract sentiment using textblob. 
        Polarity is within range [-1, 1]'''

        text = TextBlob(text)

        return text.sentiment.polarity
        