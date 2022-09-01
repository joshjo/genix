from copy import copy


MAX_TOTAL_TIME = 10
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
            ((i[1] * 0.85) / MAX_TOTAL_TIME) + ((i[2] * 0.15) / MAX_INDEX_SIZE),
        ) for i in rewards
    ]


def get_top_n(rewards, n=1):
    return sorted(rewards, key=lambda a: a[1])[:n]
