import random
import statistics
import matplotlib.pyplot as plt

# Allgemeiner Genetic Algorithm
def genetic_algorithm(fitness_fn, create_individual, crossover_fn, mutate_fn,
                      pop_size=100, crossover_rate=0.8, mutation_rate=0.1,
                      generations=500, elitism=True):
    population = [create_individual() for _ in range(pop_size)]
    best_fitness_per_gen = []

    for gen in range(generations):
        # Bewertung
        fitness_values = [fitness_fn(ind) for ind in population]
        best_fitness_per_gen.append(max(fitness_values))
        best_individual = population[fitness_values.index(max(fitness_values))]

        # Selektion
        total_fit = sum(fitness_values)
        if total_fit == 0:
            probs = [1 / len(fitness_values)] * len(fitness_values)
        else:
            probs = [f / total_fit for f in fitness_values]

        def select_parent():
            r = random.random()
            acc = 0
            for ind, p in zip(population, probs):
                acc += p
                if acc >= r:
                    return ind
            return population[-1]

        new_population = []
        if elitism:
            new_population.append(best_individual)

        # Erzeugung neue Generation
        while len(new_population) < pop_size:
            parent1 = select_parent()
            parent2 = select_parent()
            if random.random() < crossover_rate:
                child1, child2 = crossover_fn(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            if random.random() < mutation_rate:
                child1 = mutate_fn(child1)
            if random.random() < mutation_rate:
                child2 = mutate_fn(child2)
            new_population.extend([child1, child2])

        population = new_population[:pop_size]

    # Rückgabe beste Lösung
    best_fitness = max([fitness_fn(ind) for ind in population])
    best_individual = population[[fitness_fn(ind) for ind in population].index(best_fitness)]
    return best_individual, best_fitness, best_fitness_per_gen

# 8-Queens
def create_queen_individual():
    return random.sample(range(1, 9), 8)

def fitness_queens(ind):
    conflicts = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(ind[i] - ind[j]) == abs(i - j): 
                conflicts += 1
    return 28 - conflicts 

def crossover_pmx(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child1, child2 = p1[:], p2[:]
    mapping1 = dict(zip(p1[a:b], p2[a:b]))
    mapping2 = dict(zip(p2[a:b], p1[a:b]))
    for i in range(len(p1)):
        if i not in range(a, b):
            while child1[i] in mapping1:
                child1[i] = mapping1[child1[i]]
            while child2[i] in mapping2:
                child2[i] = mapping2[child2[i]]
    child1[a:b], child2[a:b] = p2[a:b], p1[a:b]
    return child1, child2

def mutate_swap(ind):
    i, j = random.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]
    return ind

# Map Coloring
neighbors = {
    "A": ["B", "C", "D"],
    "B": ["A", "C", "E"],
    "C": ["A", "B", "D", "E", "F"],
    "D": ["A", "C", "F"],
    "E": ["B", "C", "F"],
    "F": ["C", "D", "E"]
}
regions = list(neighbors.keys())
num_colors = 5
lambda_penalty = 0.5

def create_map_individual():
    return [random.randint(1, num_colors) for _ in regions]

def fitness_map(ind):
    color_map = dict(zip(regions, ind))
    conflicts = 0
    for r1 in neighbors:
        for r2 in neighbors[r1]:
            if color_map[r1] == color_map[r2]:
                conflicts += 1
    conflicts //= 2
    used_colors = len(set(ind))
    return -(conflicts + lambda_penalty * used_colors)

def crossover_one_point(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate_recolor(ind):
    idx = random.randint(0, len(ind) - 1)
    ind[idx] = random.randint(1, num_colors)
    return ind

# Testläufe
def run_experiments(problem_name, runs, create, fit, cross, mut):
    results = []
    for _ in range(runs):
        best, best_fit, fitness_curve = genetic_algorithm(
            fitness_fn=fit,
            create_individual=create,
            crossover_fn=cross,
            mutate_fn=mut,
            generations=300,
            pop_size=100
        )
        results.append(best_fit)
    print(f"\n===== {problem_name} =====")
    print(f"Durchläufe: {runs}")
    print(f"Ø Fitness: {statistics.mean(results):.2f}")
    print(f"Beste Fitness: {max(results):.2f}")
    print(f"Erfolgsrate: {(results.count(max(results)) / runs) * 100:.1f}%")

# Ausführen
run_experiments("8 Queens", 100, create_queen_individual, fitness_queens, crossover_pmx, mutate_swap)
run_experiments("Map Coloring", 100, create_map_individual, fitness_map, crossover_one_point, mutate_recolor)
