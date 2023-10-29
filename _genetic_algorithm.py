class GeneticAlgorithm:
    population_size = 10
    
    def __init__(self, genome_sample: dict) -> None:
        self.genome_sample = genome_sample
    
    def create_population(self, population_size):
        """ Create population of strategy parameters """
        return [self.generate_genome() for _ in range(population_size)]
    
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
        # # params = {key: genome[index] for index, key in enumerate(self.genome_sample.keys())}
        # params = {param_name: param_value for param_name, param_value in genome}
        # strategy.set_params(params) # Set to params to genome
        # bt = Backtest([strategy.data], strategy)
        # bt.run()
        # stats = bt.statistics()
        
        # TODO do calculation with the stats for the fitness. Direction will be given by researcher
        
        return 1_000
    
    def optimise(self, strategy: Strategy) -> [int]:
        """ Entry point to optimise strategy. Returns the best parameter genome """
        # Example
        TERMINATION_FITNESS_THRESHOLD = 10_000
        MUTATION_RATE = 0.1

        # LIMIT GENERATIONS
        MAXIMUM_GENERATION = 10

        strategy.reset()
        population = self.create_population(self.population_size)

        found = False
        generation = 0
        while not found and generation < MAXIMUM_GENERATION:
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
