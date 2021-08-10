import getheadlines
import modelv3
import yfinance as yf
import pandas as pd

file = open("tickers.txt","r")

tickers = []

for i in file:
    tickers.append(i.rstrip())

headline_sentiment = []

for i in tickers:
    company_name = yf.Ticker(i).info["longName"]
    
    headlines = getheadlines.getHeadlines(company_name)
    
    if (len(headlines) < 5):
        pass
    else:
        headlines = headlines[0:5]
    
    sentiment_score = 0
    
    for i in headlines:
        ind_sentiment = modelv3.getSentiment(i)
        sentiment_score += ind_sentiment
    
    average = sentiment_score/5
    headline_sentiment.append(average)

data = {"Ticker":tickers,"Sentiment":headline_sentiment}
df = pd.DataFrame(data)

print(df)