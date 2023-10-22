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
    
    # def generate_genome(self) -> [int]:
    #     """ Generates strategy parameters """
    #     genome = []
    #     for param in list(self.genome_sample.keys()):
    #         min_val, max_val = self.genome_sample[param]
    #         # Shouldn't I add the param here?
    #         genome.append(self.generate_randint(min_val, max_val))
        
    #     return genome
    
    def generate_genome(self):
        """ Generates strategy parameters """
        genome = []
        for param in list(self.genome_sample.keys()):
            min_val, max_val = self.genome_sample[param]
            param_value = self.generate_randint(min_val, max_val)
            genome.append((param, param_value))
        return genome
    
    def mutate(self, genome):
        """ Mutates genome by randomly changing one parameter """
        mutated_genome = genome.copy()
        # Randomly select a parameter and mutate its value
        param_index = random.randint(0, len(mutated_genome) - 1)
        param_name, param_value = mutated_genome[param_index]
        min_val, max_val = self.genome_sample[param_name]
        mutated_value = self.generate_randint(min_val, max_val)
        mutated_genome[param_index] = (param_name, mutated_value)
        return mutated_genome

    def breed(self, genome_a, genome_b):
        """ Breeds offspring from two parent genomes """
        child_genome = []
        for (param_name_a, param_value_a), (param_name_b, param_value_b) in zip(genome_a, genome_b):
            # Randomly choose parameter from parent A or B
            child_param_name = param_name_a if random.random() < 0.5 else param_name_b
            child_param_value = param_value_a if random.random() < 0.5 else param_value_b
            child_genome.append((child_param_name, child_param_value))
        return child_genome
    

    def calculate_fitness(self, strategy: Strategy, genome: [int]) -> int:
        """ Evaluates the trading strategy """
        # params = {key: genome[index] for index, key in enumerate(self.genome_sample.keys())}
        params = {param_name: param_value for param_name, param_value in genome}
        strategy.set_params(params) # Set to params to genome
        bt = Backtest(strategy.data, strategy)
        bt.run()
        stats = bt.statistics()
        
        # TODO do calculation with the stats for the fitness. Direction will be given by researcher
        
        return 1_000
    
    def optimise(self, strategy: Strategy) -> [int]:
        """ Entry point to optimise strategy. Returns the best parameter genome """
        # Example
        TERMINATION_FITNESS_THRESHOLD = 1000
        MUTATION_RATE = 0.1

        # LIMIT GENERATIONS
        MAXIMUM_GENERATION = 10

        strategy.reset()
        population = self.create_population(self.population_size)
        # self.calculate_fitness(strategy, population[0])

        found = False

        while not found:
            # Calculate fitness for each genome in the population
            fitness_scores = [self.calculate_fitness(strategy, genome) for genome in population]
            # Find the index of the best genome in the population
            best_genome_index = fitness_scores.index(max(fitness_scores))
            best_genome = population[best_genome_index]
        
            # Check if the best genome satisfies the termination condition
            if max(fitness_scores) >= TERMINATION_FITNESS_THRESHOLD:
                found = True
                break
            
            # Perform selection, crossover, and mutation to create new generation
            new_generation = []
            # Elitism: keep the best 10% of the population
            elite_count = int(0.1 * self.population_size)
            # Gets the indeces of the highest fitness scores
            elites = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k])[-elite_count:]

            # Add elites to the new generation
            new_generation.extend([population[i] for i in elites])
            
            for _ in range(self.population_size - elite_count):
                parent_a = random.choice(population)
                parent_b = random.choice(population)
                child_genome = self.breed(parent_a, parent_b)
                if random.random() < MUTATION_RATE:
                    child_genome = self.mutate(child_genome)
                new_generation.append(child_genome)
        
            population = new_generation
            generation += 1
    
        return best_genome
    
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