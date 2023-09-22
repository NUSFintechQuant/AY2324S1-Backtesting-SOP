# This file is used for testing 
from utils import *

def main():
    # 1) Sample data query
    data = {
        'Date': ['2023-09-01', '2023-09-02', '2023-09-03', '2023-09-04', '2023-09-05'],
        'Open': [150.0, 151.2, 153.5, 152.8, 152.0],
        'High': [152.3, 153.8, 155.0, 154.2, 153.5],
        'Low': [149.5, 150.7, 152.0, 151.5, 151.2],
        'Close': [151.5, 153.0, 154.5, 153.2, 152.8],
        'Volume': [100000, 120000, 95000, 110000, 105000]
    }

    # 1a) Create a DataFrame
    df = pd.DataFrame(data)

    # 1b) Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # 1c)  Set the 'Date' column as the index (optional)
    df.set_index('Date', inplace=True)

    ohlcv = [df]
    
    
    # 2) Strategy Creation
    class CrossOver(Strategy):
        def __init__(self, capital: float, commision: float, slippage: float, params: dict) -> None:
            super().__init__(capital, commision, slippage, params)
            
        def next(self):
            pass
        
    # 3) Vanilla backtest
    strategy_params = {
        "ema_period": 14,
        "rsi_period": 14,
        "upper_bound": 14,
        "lower_bound": 14,
    }         

    strategy = CrossOver(capital=10_000, commision=0.02, slippage=0.01, params=strategy_params)
    bt = Backtest(ohlcv, strategy)

    bt.run(ratio=0.7)
    bt.plot()
    bt.statistics()
    
    # 4) Event bias analysis
    bt.event_bias_analysis()

    # 5) Optimisation
    strategy_params_limit = {
        "ema_period": [5, 200],
        "rsi_period": [5, 200],
        "upper_bound": [55, 90],
        "lower_bound": [10, 45],
    }

    bt.optimise(strategy_params_limit)
    
    # 6) Walk forward Optimisation
    bt.run_walkforward(strategy_params_limit, 10, 0.7)
    bt.plot()
    bt.statistics()
    
if __name__ == "__main__":
    main()