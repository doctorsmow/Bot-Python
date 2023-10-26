from ..models.modelo_rnn import ModeloRNN
from ..services.trading_service import TradingService

class TradingBot:
    def __init__(self, api_key, secret_key):
        self.trading_service = TradingService(api_key, secret_key, 'XRP/USDT', 0.0006) # Comisión del broker en decimal (0.06% para Binance Spot Trading)
        self.model = ModeloRNN()

    def make_decision(self, data):
        prediction = self.model.predict(data)
        return prediction

    def calculate_profit(self, buy_price, sell_price, amount):
        gross_profit = (sell_price - buy_price) * amount
        net_profit = gross_profit - (2 * self.trading_service.fee_rate * gross_profit)
        return net_profit

    def run(self):
        while True:
            data = self.trading_service.get_market_data()
            decision = self.make_decision(data)
            if decision == 'buy':
                buy_order = self.trading_service.place_order('buy')
                sell_price_prediction = self.model.predict_sell_price(data)
                sell_order = self.trading_service.place_order('sell')
                profit = self.calculate_profit(buy_order['price'], sell_price_prediction, buy_order['filled'])
                if profit >= 0.1 and profit <= 0.9:  # Si la ganancia está en el rango deseado
                    print('Beneficio realizado:', profit)
                else:
                    print('Beneficio fuera del rango deseado. Ajustar estrategia de trading.')
