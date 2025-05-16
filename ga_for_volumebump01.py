import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import multiprocessing
import random
from backtesting import Backtest, Strategy
import warnings
from binance.client import Client
from datetime import datetime, timedelta
import os
import time

warnings.filterwarnings("ignore", category=RuntimeWarning)

client = Client('key', 'secret')

futures_list = [
    "ETHUSDT",      # Ethereum  
    "BNBUSDT",      # Binance Coin  
    "XRPUSDT",      # XRP  
    "ADAUSDT",      # Cardano  
    "SOLUSDT",      # Solana  
    "DOGEUSDT",     # Dogecoin  
    "DOTUSDT",      # Polkadot  
    "LTCUSDT",      # Litecoin  
    "LINKUSDT",     # Chainlink  
    "ATOMUSDT",     # Cosmos  
    "AVAXUSDT",     # Avalanche  
    "TRXUSDT",      # TRON
    "1000PEPEUSDT", # PEPE Coin
    "SUIUSDT"       # SUI Network  
]

VOLUME_THRESHOLD = 1 

def collect_futures_data():
    if not os.path.exists('futures_data'):
        os.makedirs('futures_data')
    
    start_time = "01 Feb 2025 00:00:00"
    end_time = "01 May 2025 23:59:59"

    results = []
    
    for symbol in tqdm(futures_list, desc="Downloading Futures Data"):
        try:
            klines = client.futures_historical_klines_generator(
                symbol, 
                Client.KLINE_INTERVAL_15MINUTE, 
                start_time, 
                end_time
            )
            
            filename = f"futures_data/{symbol.lower()}_15m_{datetime.now().strftime('%Y%m%d')}.txt"
            
            with open(filename, 'w') as file:
                for kline in klines:
                    # Write all 12 elements of the kline
                    data_point = ",".join(map(str, kline[:12]))
                    file.write(data_point + "\n")
            
            results.append(filename)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\nError downloading {symbol}: {str(e)}")
            continue
    
    return results

def prepare_data(filename):
    # Read all 12 kline elements
    df = pd.read_csv(filename, header=None, 
                     names=[
                         'timestamp', 'open', 'high', 'low', 'close', 'volume',
                         'close_time', 'quote_asset_volume', 'number_of_trades',
                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                         'ignore'
                     ])
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Calculate volume statistics
    df['volume_mean'] = df['volume'].rolling(window=20).mean()
    df['volume_std'] = df['volume'].rolling(window=20).std()
    df['volume_z_score'] = (df['volume'] - df['volume_mean']) / df['volume_std']
    
    # Calculate buy ratio and order type
    df['buy_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
    df['order_type'] = df['buy_ratio'].apply(
        lambda x: 'BUY' if x > 0.6 else ('SELL' if x < 0.4 else 'MIXED')
    )
    
    # Identify significant volume events
    significant_volume = (df['volume_z_score'] > 2) & (df['volume'] > VOLUME_THRESHOLD)
    df['large_buy'] = (df['order_type'] == 'BUY') & significant_volume
    df['large_sell'] = (df['order_type'] == 'SELL') & significant_volume
    
    return df

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def initialize_ga():
    toolbox = base.Toolbox()
    
    def generate_sl():
        return round(random.uniform(0.6, 3.50), 2)  
    
    def generate_tp():
        return round(random.uniform(0.6, 3.50), 2)  
    
    def create_individual():
        return [
            generate_sl(),   
            generate_tp(),    
        ]

    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    return toolbox

def mutate_individual(individual):
    mutation_rate = 0.2
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            change = random.uniform(-0.02, 0.02)
            new_value = individual[i] + change
            individual[i] = round(max(0.60, min(3.50, new_value)), 2)
    return individual,

def evaluate_strategy(individual, df, initial_cash=10000, commission=0.00075):
    try:
        class LargeOrderStrategy(Strategy):
            def init(self):
                self.stop_loss = individual[0]
                self.take_profit = individual[1]

            def next(self):
                current_close = self.data.Close[-1]
                
                if self.data.large_buy[-1]:
                    if self.position.is_short:
                        self.position.close()
                    self.buy(
                        sl=current_close * (1 - self.stop_loss/100),
                        tp=current_close * (1 + self.take_profit/100)
                    )

                elif self.data.large_sell[-1]:
                    if self.position.is_long:
                        self.position.close()
                    self.sell(
                        sl=current_close * (1 + self.stop_loss/100),
                        tp=current_close * (1 - self.take_profit/100)
                    )

        # Prepare backtest data with proper column names
        bt_data = df.copy()
        bt_data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
        }, inplace=True)
        bt_data.set_index('timestamp', inplace=True)

        bt = Backtest(bt_data, LargeOrderStrategy,
                     cash=initial_cash,
                     commission=commission,
                     exclusive_orders=True)
        
        stats = bt.run()
        total_return = stats['Return [%]']
        win_rate = stats['Win Rate [%]']
        num_trades = stats['# Trades']
        
        if num_trades < 2:
            return (-999999.0,)
        
        fitness = total_return
        
        if win_rate < 45:
            fitness *= 0.8
        
        return (fitness,)
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return (-999999.0,)

# Rest of the code remains the same as original (optimize_single_symbol and main)

def optimize_single_symbol(filename):
    df = prepare_data(filename)
    symbol = filename.split('/')[-1].split('_')[0].upper()
    
    toolbox = initialize_ga()
    
    toolbox.register("evaluate", evaluate_strategy, df=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    pop_size = 85
    n_generations = 50
    population = toolbox.population(n=pop_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    print(f"\nOptimizing strategy for {symbol}")
    
    for gen in tqdm(range(n_generations), desc=f"Evolving {symbol}"):
        algorithms.eaSimple(population, toolbox,
                          cxpb=0.7,
                          mutpb=0.2,
                          ngen=1,
                          stats=stats,
                          verbose=False)
    
    best_strategy = tools.selBest(population, k=1)[0]
    
    # Final backtest with best parameters
    bt_data = df.copy()
    bt_data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)
    bt_data.set_index('timestamp', inplace=True)

    class FinalStrategy(Strategy):
        def init(self):
            self.stop_loss = best_strategy[0]
            self.take_profit = best_strategy[1]
        
        def next(self):
            current_close = self.data.Close[-1]
            if self.data.large_buy[-1]:
                if self.position.is_short:
                    self.position.close()
                self.buy(
                    sl=current_close * (1 - self.stop_loss/100),
                    tp=current_close * (1 + self.take_profit/100)
                )
            elif self.data.large_sell[-1]:
                if self.position.is_long:
                    self.position.close()
                self.sell(
                    sl=current_close * (1 + self.stop_loss/100),
                    tp=current_close * (1 - self.take_profit/100)
                )

    bt = Backtest(bt_data, FinalStrategy,
                 cash=10000,
                 commission=0.00075,
                 exclusive_orders=True)
    
    stats = bt.run()
    
    pool.close()
    
    return {
        'symbol': symbol,
        'stop_loss': best_strategy[0],
        'take_profit': best_strategy[1],
        'return': stats['Return [%]'],
        'stats': stats
    }

def main():
    print("Starting data collection...")
    data_files = collect_futures_data()
    
    all_results = []
    
    for file in data_files:
        try:
            result = optimize_single_symbol(file)
            if result['return'] > -100:  
                all_results.append(result)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    all_results.sort(key=lambda x: x['return'], reverse=True)
    
    print("\n=== Strategy Results ===")
    for result in all_results:
        print(f"\nSymbol: {result['symbol']}")
        print(f"Stop Loss: {result['stop_loss']}%")
        print(f"Take Profit: {result['take_profit']}%")
        print("-" * 40)
        print("\nPerformance Metrics:")
        print(result['stats'])
        print("=" * 40)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStrategy optimization interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nStrategy optimization completed")
