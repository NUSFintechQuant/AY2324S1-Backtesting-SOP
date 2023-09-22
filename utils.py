import random
import pandas as pd
from enum import Enum
from typing import Optional

class Indicator:
    # TODO To be implemented by Justin
    pass

class Side(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    
class Trade:
    trade_id = 0
    side: Side
    datetime: pd.Timestamp
    entry_price: float
    exit_price: float
    Size: float
    is_filled: bool
    is_closed: bool
    
    def __init__(self, side: Side,  datetime: pd.Timestamp,  entry_price: float, size: float) -> None:
        trade_id = Trade.trade_id
        Trade.trade_id += 1
        self.side = side
        self.datetime = datetime
        self.entry_price = entry_price
        self.exit_price = None
        self.size = size
        self.is_filled = False
        self.is_closed = False
        
    def pnl(self) -> float:
        pass

    def is_long(self) -> bool:
        pass

    def is_short(self) -> bool:
        pass
    
    def is_closed(self) -> bool:
        pass

    def close(self, portion: float = 1.0):
        pass

    def get_entry_price(self) -> float:
        pass

    def get_exit_price(self) -> float:
        pass

    def get_get_size(self) -> float:
        pass

    def is_filled(self) -> bool:
        pass

class _Exchange:
    date_index: int = 0
    _opened_trades: [Trade]
    _closed_trades: [Trade]
    
    def __init__(self, capital: float, commision: float, slippage: float) -> None:
        self.capital = capital
        self.commision = commision
        self.slippage = slippage
        self._opened_trades = []
        self._closed_trades = []
        self.data = None

    def set_data(self, data) -> None:
        self.data

    def next(self) -> None:
        pass

    def increment_date_index(self) -> None:
        self.date_index += 1

    def buy(self, id: int, price: float, size: float) -> None:
        datetime = self.data[self.date_index]
        trade = Trade(Side.LONG, datetime, price, size)
        self._opened_trades.append(trade)
        pass

    def sell(self, id: int, price, size: float) -> None:
        pass

    def positions(self) -> [Trade]:
        pass

    def trades(self) -> Trade:
        pass

    def reset(self) -> None:
        pass

    def statistics(self) -> None:
        pass

    
class Strategy:
    def __init__(self, capital: float, commision: float, slippage: float, params: dict) -> None:
        self._exchange = _Exchange(capital, commision, slippage)
        self.params = params
    
    def set_data(self, data):
        self._exchange.set_data(data)
        
    def set_params(self, params):
        self.params = params
        
    def next(self):
        # Implement your strategy logic here
        pass
    
    def increment_date_index(self):
        self._exchange.increment_date_index()

    def buy(self, price, size):
        """ Enters a long trade """
        # Stores trade into _opened_trades
        self._exchange.buy(price, size)

    def sell(self, price, size):
        """ Enters a short trade """
        # Stores trade into _closed_trades
        self._exchange.sell(price, size)

    def positions(self):
        # Implement logic to retrieve current positions
        
        pass

    def trades(self):
        # Implement logic to retrieve trade history
        pass
    
    def reset(self):
        # Implement reset all parameters and fields of strategy
        pass
    
    def statistics(self):
        # Implement return of statistics
        pass

class GeneticAlgorithm:
    population_size = 1000
    
    def __init__(self, genome_sample: dict) -> None:
        self.genome_sample = genome_sample
    
    def create_population(self, population_size):
        """ Create population of strategy parameters """
        return [self.generate_genome() for _ in range(population_size)]
    
    def generate_genome(self) -> [int]:
        """ Generates strategy parameters """
        genome = []
        for param in list(self.genome_sample.keys()):
            min_val, max_val = self.genome_sample[param]
            genome.append(self.generate_randint(min_val, max_val))
        
        return genome
    
    def mutate(self, genome) -> [int]:
        """ Mutates genome """
        pass
    
    def breed(self, genome_a, genome_b) -> [int]:
        """ Breeds offspring from 2 genome """
        pass
    
    def calculate_fitness(self, strategy: Strategy, genome: [int]) -> int:
        """ Evaluates the trading strategy """
        params = {key: genome[index] for index, key in enumerate(self.genome_sample.keys())}
        strategy.set_params(params) # Set to params to genome
        bt = Backtest(strategy.data, strategy)
        bt.run()
        stats = bt.statistics()
        
        # TODO do calculation with the stats for the fitness. Direction will be given by researcher
        
        return 1_000
    
    def optimise(self, strategy: Strategy) -> [int]:
        """ Entry point to optimise strategy. Returns the best parameter genome """
        # Example
        strategy.reset()
        population = self.create_population(self.population_size)
        self.calculate_fitness(strategy, population[0])
        
        # TODO Devan Pseudocode
        # POPULATION_SIZE
        # population = []
        
        # found = False
        
        
        # while not found:
        #     # sort the population in increasing order of fitness score
        #     population = sorted(population, key = lambda x:x.fitness)
    
        #     # if the individual having lowest fitness score ie. 
        #     # 0 then we know that we have reached to the target
        #     # and break the loop
        #     if population[0].fitness <= 0:
        #         found = True
        #         break
    
        #     # Otherwise generate new offsprings for new generation
        #     new_generation = []
    
        #     # Perform Elitism, that mean 10% of fittest population
        #     # goes to the next generation
        #     s = int((10*POPULATION_SIZE)/100)
        #     new_generation.extend(population[:s])
    
        #     # From 50% of fittest population, Individuals 
        #     # will MATE to produce offspring
        #     s = int((90*POPULATION_SIZE)/100)
        #     for _ in range(s):
        #         parent1 = random.choice(population[:50])
        #         parent2 = random.choice(population[:50])
        #         child = parent1.mate(parent2)
        #         new_generation.append(child)
    
        #     population = new_generation
    
        #     generation += 1
        pass
    
    def generate_randint(self, min, max) -> int:
        """ Generates random parameters """
        return random.randint(min, max)

class Backtest:
    def __init__(self, data: [pd.DataFrame], strategy: Strategy) -> None:
        self.data = data
        self.strategy: Strategy = strategy
        self.strategy.set_data(data[0])
        
    def run(self, ratio=0.7):
        """ Vanilla backtest """
        # TODO Below is just an example of how it should run.
        # I recognise that there is an issue with the splitting of df, do fix that.
        for data in self.data:
            self._reset_strategy()

            data_size = len(self.data)
            train_window = data_size * ratio
            train_data = data[:train_window]
            self.strategy.set_data(train_data)
            for dateIndex, d in enumerate(train_data): # Train
                self.strategy.next()
                self.strategy.increment_date_index()
            
            # TODO Get the parameters from train and pass to test.
            # Store the train and test result because you need to plot them
            
            self._reset_strategy()
            test_data = data[train_window:]
            self.strategy.set_data(test_data)
            for dateIndex, d in enumerate(test_data): # Test
                self.strategy.next()
                self.strategy.increment_date_index()

    def optimise(self, strategy_params_limit):
        """ Vanilla backtest with 1 run of optimisation """
        self._reset_strategy()
        
        GeneticAlgorithm(strategy_params_limit).optimise(self.strategy)

    def run_walkforward(self, strategy_params_limit, window_count, ratio):
        """ Walk forward optimisation """
        self._reset_strategy()
        
        self.optimise(strategy_params_limit)

        pass
    
    def event_bias_analysis(self):
        pass

    def plot(self):
        """ Plots the PnL chart """
        pass

    def statistics(self):
        """ Displays the statistics of the algorithm """
        # TODO take from strategy and display
        pass
    
    def _reset_strategy(self):
        self.strategy.reset()
        pass