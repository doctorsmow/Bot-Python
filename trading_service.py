import ccxt

class TradingService:
    def __init__(self, api_key, secret_key, trade_symbol, fee_rate):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key
        })
        self.trade_symbol = trade_symbol
        self.fee_rate = fee_rate
        self.initial_order_amount = 10  # 10 USDT for the first buy order
        self.balance = 0

    def get_market_data(self):
        ticker = self.exchange.fetch_ticker(self.trade_symbol)
        return ticker

    def place_order(self, order_type):
        amount = self.initial_order_amount if self.balance < 100 else self.balance * 0.1
        if order_type == 'buy':
            order = self.exchange.create_market_buy_order(self.trade_symbol, amount)
        elif order_type == 'sell':
            order = self.exchange.create_market_sell_order(self.trade_symbol, amount)
        self.update_balance(order)
        return order

    def update_balance(self, order):
        if order['side'] == 'buy':
            self.balance -= order['cost']
        elif order['side'] == 'sell':
            self.balance += order['cost']
