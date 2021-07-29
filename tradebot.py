"""
mean reversion trading bot
main trade algorithm: includes live and historical testing
created and developed by: luke marolda, carnegie mellon university
""" 

import json, requests, websocket
import numpy as np
from config import *

### GLOBALS ###

CURRENT_SUBSCRIPTION = ""
CURRENT_STOCK = ""


### STOCK CLASS ###

class Stock(object):
    # initialize stock object with symbol, historical and live info
    def __init__(self, symbol):
        self.symbol = symbol
        self.in_position = False
        self.positionSize = ""
        self.positionType = ""
        self.boughtFor = 0
        self.livePrices = []
        self.percentageChanges = []

    # function executes a market order
    def create_market_order(self, side, order_type, qty, time_in_force):
        data = {
            "side": side,
            "symbol": self.symbol,
            "type": order_type,
            "qty": qty,
            "time_in_force": time_in_force,
        }
        r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
        return json.loads(r.content)    

    # function executes an oto order with only a stop loss
    def create_oto_order(self, side, order_type, qty, time_in_force, stop_price):
        data = {
            "side": side,
            "symbol": self.symbol,
            "type": order_type,
            "qty": qty,
            "time_in_force": time_in_force,
            "order_class": "oto",
            "stop_loss": {
                "stop_price": stop_price,
            }
        }
        r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
        return json.loads(r.content)

    # function to retrieve historical closes of stock object
    # note startDate and endDate must be in RFC-3339 format
    def get_historical_closes(self, startDate, endDate, limit, timeframe):
        # initialize our result list
        listOfCloses = []
        # build URL endpoint based on inputs
        symbol_url = "https://data.alpaca.markets/v2/stocks/{}/bars".format(self.symbol)
        start_param = "?start={}".format(startDate)
        end_param = "&end={}".format(endDate)
        limit_param = "&limit={}".format(limit)
        timeframe_param = "&timeframe={}".format(timeframe)
        quote_url = symbol_url + start_param + end_param + limit_param + timeframe_param
        # make request for bars
        r = requests.get(quote_url, headers=HEADERS)
        # make response a workable json dict
        data = r.json()
        numberOfBars = len(data["bars"])
        #print(data["bars"][0])
        for bar in range(0, numberOfBars):
            currentBar = data["bars"][bar]
            currentClose = currentBar["c"]
            listOfCloses.append(currentClose)
        return listOfCloses


### ALPACA ACCOUNT INFO ###

# function loads account data
def get_account():
    r = requests.get(ACCOUNT_URL, headers=HEADERS)
    return json.loads(r.content)

# function returns all active orders
def get_orders():
    r = requests.get(ORDERS_URL, headers=HEADERS)
    return json.loads(r.content)

# function returns the current quote of a given stock
def get_quote(ticker):
    symbol_url = "https://data.alpaca.markets/v2/stocks/{}/quotes".format(ticker)
    quote_url = symbol_url + "?start=2021-06-01T12:00:00.000Z" + "&end=2021-07-01T12:00:00.000Z"
    r = requests.get(quote_url, headers=HEADERS)
    return json.loads(r.content)


### TRADING ALGO ###

# script for simple linear regression
def find_regression_coef(x, y):
    n = np.size(x)
    # mean of x and y
    m_x = np.mean(x)
    m_y = np.mean(y)
    # find cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    # safety 
    if (SS_xx == 0):
        return (0,0)
    # calculating regression coefficients
    b1 = SS_xy / SS_xx
    b0 = m_y - b1 * m_x
    # where line of best fit is y = b1 * x + b0
    return (b0, b1)

# function performs durbin watson statistic on list of price percent changes
def durbinWatson(percentageChanges):
    # how long of list we use for serial correl
    serialLen = 10
    length = len(percentageChanges)
    # ensure there is enough data to perform statistic
    if (length < serialLen + 1):
        return
    startPoint = length - serialLen - 1
    # create the two working lists we will use for regression
    laggedData = percentageChanges[startPoint: startPoint + serialLen]
    liveData = percentageChanges[startPoint + 1: startPoint + 1 + serialLen]
    # convert to numpy array
    (x,y) = (np.array(laggedData), np.array(liveData))
    # calculate coeffs
    (b0, b1) = find_regression_coef(x, y)
    expValues = []
    # calculate the expected y values for laggedData using eqn
    for x in laggedData:
        expValue = (b1 * x) + b0
        expValues.append(expValue)
    residuals = []
    sumOfResidualSqr = 0
    # calculate the residuals
    for i in range(serialLen):
        residual = liveData[i] - expValues[i]
        residuals.append(residual)
        # square and add to sum of residuals
        sumOfResidualSqr += (residual ** 2)
    # now find the sum of differences in residuals squared
    sumOfDifferenceSqr = 0
    for i in range(1, serialLen):
        difference = residuals[i] - residuals[i - 1]
        sumOfDifferenceSqr += (difference ** 2)
    # if sum of residuals squared is 0, avoid error
    if (sumOfResidualSqr == 0):
        return 4
    durbinWatson = sumOfDifferenceSqr / sumOfResidualSqr
    return durbinWatson

# given a stock, calculates the sma value for a specified period
def sma(stock, period):
    # SMA at period N = (A1 + A2 + A3 + AN) / N 
    # where N is the total number of periods
    sumOfCloses = 0.0
    startingDay = period
    totalDays = len(stock)
    if (period > totalDays):
        return
    for day in range(totalDays - period, totalDays):
        close = stock[day]
        sumOfCloses += close
    sma = sumOfCloses / period
    return sma

# returns a tuple of the lower and upper bollinger bands for a stock
def bollingerBands(stock, mult):
    length = len(stock)
    period = 20
    # safety for sma
    if (length < period):
        return
    # separate the stock data we want stdev for
    data = stock[length - period: length]
    # base line is 20-period sma
    base = sma(stock, 20)
    # convert stock closes to numpy array
    closes = np.array(data)
    stdev = np.std(closes)
    # calculate the bands
    bbLower = base - (mult * stdev)
    bbUpper = base + (mult * stdev)
    return (bbLower, bbUpper)

# this function tests the trading strategy on historical price data
def testTradingAlgo(currentStock):
    numOfCloses = "500"
    closes = currentStock.get_historical_closes("2021-06-01T12:00:00.000Z", 
             "2021-07-01T12:00:00.000Z", numOfCloses, "1Min")
    percentageChanges = [0]
    for i in range(1, int(numOfCloses)):
        (price, prevPrice) = (closes[i], closes[i - 1])
        percentChange = ((price - prevPrice) / prevPrice) * 100
        percentageChanges.append(percentChange)

    ### SIMULATION ###
    in_position = False
    positionType = ""
    cash = 100000
    shares = 0
    current_position = 0
    boughtFor = 0
    # iterate through the data as if real time price data
    for i in range(1, int(numOfCloses) + 1):
        prices = closes[0:i]
        percentages = percentageChanges[0:i]
        closePrice = prices[-1]
        # need 30 min of candlestick data before executing
        if (len(prices) < 30):
            continue
        # now we can execute trades based on our strategy, given we have enough data
        dw = durbinWatson(percentages)
        bb = bollingerBands(prices, 2)
        mean = sma(prices, 20)
        # calculate account info
        current_position = closePrice * shares
        # if the market has slight negative or no serial correlation (non-trending)
        if (not in_position) and (1.5 <= dw and dw < 2.0):
            tradeQty = 500
            # if we close above the upper bollinger band
            if (closePrice > bb[1]):
                # then we want to short the stock, and buy back at the mean
                current_position = closePrice * tradeQty
                boughtFor = closePrice
                cash += current_position
                shares -= tradeQty
                # update quantity and position attributes
                in_position = True
                positionType = "short"
                print("shorted stock for {}".format(closePrice))
            # if we close below the lower bollinger band
            elif (closePrice < bb[0]):
                # then we want to buy the stock, and sell back at the mean
                current_position = closePrice * tradeQty
                boughtFor = closePrice
                cash -= current_position
                shares += tradeQty
                # update quantity and position attributes
                in_position = True
                positionType = "long"
                print("bought stock for {}".format(closePrice))
        # if we are in a position, search for trade exit
        else:
            qty = abs(shares)
            percentDiff = boughtFor * 0.01
            # stop losses
            if (positionType == "short") and ((closePrice - boughtFor) > percentDiff):
                current_position = closePrice * qty
                cash -= current_position
                shares += qty
                in_position = False
                positionType = ""
                boughtFor = 0
                print("hit stop loss at {}".format(closePrice))
            elif (positionType == "long") and ((boughtFor - closePrice) > percentDiff):
                current_position = closePrice * qty
                cash += current_position
                shares -= qty            
                in_position = False
                positionType = ""
                boughtFor = 0
                print("hit stop loss at {}".format(closePrice))
            # take profits
            # if we are in a short position, look to buy back
            elif (positionType == "short") and (closePrice <= mean):
                current_position = closePrice * qty
                cash -= current_position
                shares += qty
                in_position = False
                positionType = ""
                boughtFor = 0
                print("bought back stock for {}".format(closePrice))
            # if we are in a long position, look to sell back
            elif (positionType == "long") and (closePrice >= mean):
                current_position = closePrice * qty
                cash += current_position
                shares -= qty            
                in_position = False
                positionType = ""  
                boughtFor = 0
                print("sold back stock for {}".format(closePrice))
    # calculate results
    portfolio = cash + current_position
    return portfolio


### HISTORICAL TESTING ###

# pulled variety of stocks from dow jones, and tested trading strategy on this
# past month's (june) price action data. 

"""
stocks = ["AXP", "AAPL", "BA", "CSCO", "GS", "HD", "HON", "IBM",
          "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT", 
          "NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", 
          "DIS"]
# now lets test!
portfolios = []
for ticker in stocks:
    stock = Stock(ticker)
    portfolio = {ticker : testTradingAlgo(stock)}
    portfolios.append(portfolio)
print(portfolios)
"""


### LIVE MARKET DATA ###

# determine which stock we will be streaming live data for and trading
CURRENT_STOCK = Stock("AAPL")
CURRENT_SUBSCRIPTION = CURRENT_STOCK.symbol

# when we open the websocket,
def on_open(ws):
    print("opened")
    # data message to authenticate self
    auth_data = {
        "action": "auth",
        "key": API_KEY,
        "secret": SECRET_KEY
    }
    # send message, converting from python dict to json string
    ws.send(json.dumps(auth_data))
    # send message to determine what we want to listen for
    listen = {
        "action": "subscribe",
        "bars": [CURRENT_SUBSCRIPTION],
    }
    ws.send(json.dumps(listen))

# function to execute trading strategy every time a quote is recieved
def on_message(ws, message):
    global CURRENT_STOCK
    # parse payload
    data = json.loads(message)
    # retrieve first index of message, which is quote
    currentBar = data[0]
    # obtain open and close data for bar
    openPrice = currentBar["o"]
    closePrice = currentBar["c"]
    # calculate percentage change of each minute bar
    percentageChange = ((closePrice - openPrice) / openPrice) * 100
    CURRENT_STOCK.percentageChanges.append(percentageChange)
    CURRENT_STOCK.livePrices.append(closePrice)
    # need 30 min of candlestick data before executing
    if (len(CURRENT_STOCK.livePrices) < 30):
        return
    if (len(CURRENT_STOCK.livePrices) == 30):
        print("## Now beginning trade strategy execution ##")
    # now we can execute trades based on our strategy, given we have enough data
    dw = durbinWatson(CURRENT_STOCK.percentageChanges)
    bb = bollingerBands(CURRENT_STOCK.livePrices, 2)
    mean = sma(CURRENT_STOCK.livePrices, 20)
    # if the market has slight negative or no serial correlation (non-trending)
    if (not CURRENT_STOCK.in_position) and (1.5 <= dw and dw < 2):
        stopPercentage = 0.005
        tradeQty = "100"
        # if we close above the upper bollinger band
        if (closePrice > bb[1]):
            # then we want to short the stock, and buy back at the mean
            stop_price = str(closePrice + (closePrice * stopPercentage))
            CURRENT_STOCK.create_market_order("sell", "market", tradeQty, "gtc")
            print("## Trade Signaled: Shorted stock with market order ##")
            # update quantity and position attributes
            CURRENT_STOCK.positionSize = tradeQty
            CURRENT_STOCK.in_position = True
            CURRENT_STOCK.positionType = "short"
            CURRENT_STOCK.boughtFor = closePrice
        # if we close below the lower bollinger band
        elif (closePrice < bb[0]):
            # then we want to buy the stock, and sell back at the mean
            stop_price = str(closePrice - (closePrice * stopPercentage))
            CURRENT_STOCK.create_market_order("buy", "market", tradeQty, "gtc")
            print("## Trade Signaled: Longed stock with market order ##")
            # update quantity and position attributes
            CURRENT_STOCK.positionSize = tradeQty
            CURRENT_STOCK.in_position = True
            CURRENT_STOCK.positionType = "long"
            CURRENT_STOCK.boughtFor = closePrice
    # if we are in a position, search for trade exit
    else:
        size = CURRENT_STOCK.positionSize
        qty = int(size)
        percentDiff = CURRENT_STOCK.boughtFor * 0.005
        # stop losses
        if (CURRENT_STOCK.positionType == "short") and ((closePrice - CURRENT_STOCK.boughtFor) > percentDiff):
            CURRENT_STOCK.create_market_order("buy", "market", size, "gtc")
            print("## Stop Loss Hit: Bought back stock for a loss ##")
            CURRENT_STOCK.positionSize = ""
            CURRENT_STOCK.in_position = False
            CURRENT_STOCK.positionType = ""
            CURRENT_STOCK.boughtFor = 0
        elif (CURRENT_STOCK.positionType == "long") and ((CURRENT_STOCK.boughtFor - closePrice) > percentDiff):
            CURRENT_STOCK.create_market_order("sell", "market", size, "gtc")
            print("## Stop Loss Hit: Sold back stock for a loss ##")
            CURRENT_STOCK.positionSize = ""
            CURRENT_STOCK.in_position = False
            CURRENT_STOCK.positionType = ""
            CURRENT_STOCK.boughtFor = 0
        # take profits
        # if we are in a short position, look to buy back
        elif (CURRENT_STOCK.positionType == "short") and (closePrice <= mean):
            CURRENT_STOCK.create_market_order("buy", "market", size, "gtc")
            print("## Exit Found: Bought back stock with market order ##")
            CURRENT_STOCK.positionSize = ""
            CURRENT_STOCK.in_position = False
            CURRENT_STOCK.positionType = ""
            CURRENT_STOCK.boughtFor = 0
        # if we are in a long position, look to sell back
        elif (CURRENT_STOCK.positionType == "long") and (closePrice >= mean):
            create_market_order("sell", "market", size, "gtc")
            print("## Exit Found: Sold back stock with market order ##")
            CURRENT_STOCK.positionSize = ""
            CURRENT_STOCK.in_position = False
            CURRENT_STOCK.positionType = ""
            CURRENT_STOCK.boughtFor = 0
    
def on_close(ws):
    print("connection closed")

# function begins live quotes
def start_live_quotes():
    ws.run_forever()


### INITIATE LIVE DATA STREAM ###

# create a new instance of websocket app
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_close=on_close)
# begin live quotes
start_live_quotes()
