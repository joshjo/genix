import math

from copy import copy, deepcopy
from random import randint, random


MAX_TOTAL_TIME = 100
MAX_INDEX_SIZE = 100_000_000


def is_dominant(a, b):
    return a[1] <= b[1] and a[2] <= b[2]


def get_pareto(values):
    pareto = copy(values)
    i = 0
    while i < len(pareto):
        j = i + 1
        while j < len(pareto):
            if is_dominant(pareto[i], pareto[j]):
                del pareto[j]
                continue
            j += 1
        i += 1
    return pareto


def normalize_rewards(rewards):
    return [
        (
            i[0],
            ((i[1] * 0.7) / MAX_TOTAL_TIME) + ((i[2] * 0.3) / MAX_INDEX_SIZE),
        ) for i in rewards
    ]


def aggregate_equation(x):
    """
    - value 0: 0.7 execution time,
    - value 1: 0.3
    """
    return ((x[0] * 0.8) / MAX_TOTAL_TIME) + ((x[1] * 0.2) / MAX_INDEX_SIZE)


def _get_non_dominants(values):
    pareto = copy(values)
    pareto.sort(key=lambda x: x[1])
    i = 0
    while i < len(pareto):
        j = i + 1
        while j < len(pareto):
            a, b = pareto[i][1], pareto[j][1]
            # if is dominant
            if all([ai <= bi for ai, bi in zip(a, b)]):
                del pareto[j]
                continue
            j += 1
        i += 1
    return pareto


def set_elitism_multi_objective(population):
    raw_population = [(elem.process_id, elem._fitness) for elem in population]
    pareto_ids = [elem[0] for elem in _get_non_dominants(raw_population)]
    for elem in population:
        elem.is_elite = elem.process_id in pareto_ids


def set_elitism_single_objective(population):
    top_n = math.ceil(len(population) * 0.075)
    population.sort(key=lambda x: x.fitness)
    for i, elem in enumerate(population):
        elem.is_elite = i < top_n


def evolve_deme(deme, is_multi_objective, runner, metropolis, options={}):
    """
    population: List of BaseGen.
    """
    population = deme.population
    len_population = len(population)
    # Copy elite members
    new_population = [elem for elem in population if elem.is_elite]
    # Copy some elements from immigrants
    for immigrant in deme.immigrants:
        if random() < 0.05:
            immigrant.elem.is_elite = False
            new_population.append(immigrant.elem)

    while len(new_population) < len_population:
        new_elem_1 = population[randint(0, len_population - 1)]
        new_elem_2 = population[randint(0, len_population - 1)]
        new_elem = deepcopy(
            new_elem_1 if new_elem_1.is_dominant(new_elem_2) else new_elem_2
        )
        new_elem.assign_process_id()
        new_elem.is_elite = False
        new_population.append(new_elem)

    for elem in new_population:
        if elem.is_elite and not random() < metropolis:
            continue
        if random() < 0.3:
            elem.mutate()
        if random() < 0.7:
            rindex = randint(0, len_population - 1)
            elem.crossover(
                new_population[rindex],
                partial=new_population[rindex].is_elite,
            )
        if random() < 0.1 and deme.immigrants:
            rindex = randint(0, len(deme.immigrants) - 1)
            elem.extend(deme.immigrants[rindex].elem)


    for elem in new_population:
        elem._fitness = runner(elem=elem, **options)

    if is_multi_objective:
        set_elitism_multi_objective(new_population)
    else:
        set_elitism_single_objective(new_population)

    return deme.name, new_population
