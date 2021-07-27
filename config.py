# mean reversion trading bot
# api configuration

# api key id and secret key
API_KEY = "YOUR_ID"
SECRET_KEY = "YOUR_KEY"

# endpoint URLs
BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
ORDERS_URL = "{}/v2/orders".format(BASE_URL)
DATA_URL = "https://data.alpaca.markets/v2"

# headers for requests
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

# this is our socket
SOCKET = "wss://stream.data.alpaca.markets/v2/iex"

# other
APCA_RETRY_MAX = 3
APCA_RETRY_CODES = 429,504




