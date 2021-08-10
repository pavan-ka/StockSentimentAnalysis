from pygooglenews import GoogleNews

def getHeadlines(ticker):
    news = GoogleNews()

    stock_headlines = news.search(ticker)
    headlines = []

    for i in stock_headlines['entries']:
        headlines.append(i['title'])
    
    return headlines