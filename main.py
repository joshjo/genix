from collections import defaultdict, namedtuple
from copy import deepcopy
from random import random, randint
from dotenv import dotenv_values
import abc
import math
import multiprocessing as mp
import argparse
from tensorboardX import SummaryWriter
import uuid
from db_helper import (
    BENCHMARK_QUERIES,
    TEST_QUERIES,
    get_defs,
    run_index,
)
from ga_helper import evolve_deme
from local_cache import LocalCache
from plot_helper import plot_generation


parser = argparse.ArgumentParser(description='GENIX runner.')
parser.add_argument(
    '--debug', default=False, action=argparse.BooleanOptionalAction)

ACTIVE = 1
NOINDEX = 0

MAX_TOTAL_TIME = 100
MAX_INDEX_SIZE = 100_000_000
FITNESS_LIMTIS = [MAX_TOTAL_TIME, MAX_INDEX_SIZE]

BIDIRECTIONAL = 0
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = 2

# ENV_VALUES = dotenv_values(".env")
# is_debug = ENV_VALUES.get("DEBUG", "False").lower() in ["true", "1"]


# The relation is based is on who received the connection
edges = {
    'NATION': ['REGION'],
    'SUPPLIER': ['NATION'],
    'PARTSUPP': ['PART', 'SUPPLIER'],
    'CUSTOMER': ['NATION'],
    'ORDERS': ['CUSTOMER'],
    'LINEITEM': ['ORDERS', 'PARTSUPP'],
}


Immigrant = namedtuple("Immigrant", ["elem", "source"])


def get_log_dir(main_folder, comment):
    import socket
    import os
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    return os.path.join(
        main_folder, current_time + '_' + socket.gethostname() + comment)


class BaseGen(abc.ABC):
    def __init__(self, index_defs, initial_fitness=[0], fitness_weights=[]):
        self.gen = [NOINDEX for _ in index_defs]
        self.index_defs = index_defs
        self.assign_process_id()
        self._fitness = initial_fitness
        self.is_elite = False
        self.fitness_weights = fitness_weights

    def pick_best(self, other, ratio=0.5):
        for i in range(len(self.gen)):
            if other.gen[i] == ACTIVE and random() < ratio:
                self.gen[i] = other.gen[i]

    @property
    # @abstractmethod
    def fitness(self):
        return self._fitness

    def mutate(self, ratio=0.2):
        """
        The chance of removing an index should be greater than adding
        """
        rindex = randint(0, len(self.gen) - 1)
        if self.gen[rindex] == NOINDEX and random() < ratio:
          self.gen[rindex] = ACTIVE
        elif self.gen[rindex] == ACTIVE and random() < 1 - ratio:
            self.gen[rindex] = 0

    def is_dominant(self, other):
        return all([a < b for a, b in zip(self.fitness, other.fitness)])

    def crossover(self, other, partial=False):
        rindex = randint(0, len(self.gen) - 2)
        if partial:
            self.gen[:rindex] = other.gen[:rindex]
        else:
            self.gen[:rindex], other.gen[:rindex] = other.gen[:rindex], self.gen[:rindex]

    def get_gen_str(self):
        array_str = ["█" if i else "·" for i in self.gen]
        return "".join(array_str)

    def assign_process_id(self):
        self.process_id = str(uuid.uuid4())[:8]

    def get_phenotype(self):
        indexes = [v for k, v in zip(self.gen, self.index_defs) if k]
        table_columns = defaultdict(list)
        for k, v in indexes:
            table_columns[k].append(v)
        triplets = []
        for table, columns in table_columns.items():
            str_columns = ", ".join(columns)
            index_name = "__".join(columns)
            index_name = f"{table}__{index_name}"
            triplets.append((index_name, table, str_columns))

        return triplets

    def get_phenotype_queries(self, phenotype=None):
        if phenotype is None:
            phenotype = self.get_phenotype()
        return [
            f"CREATE INDEX IF NOT EXISTS {indexname} ON {table}({columns});"
            for indexname, table, columns in phenotype
        ]


class RandomTableGen(BaseGen):
    def __init__(self, index_defs, initial_fitness=[0], ratio=0.5, from_pos=0, length=0, fitness_weights=[]):
        super().__init__(index_defs, initial_fitness, fitness_weights)
        size = len(index_defs)
        to_pos = min(from_pos + length, size)
        for i in range(from_pos, to_pos):
            self.gen[i] = ACTIVE if random() < ratio else NOINDEX
        self.from_pos = from_pos
        self.to_pos = to_pos

    def extend(self, other):
        self.gen[other.from_pos:other.to_pos] = other.gen[other.from_pos:other.to_pos]

    def mutate(self, ratio=0.2):
        """
        The chance of removing an index should be greater than adding
        """
        rindex = randint(self.from_pos, self.to_pos - 1)
        if self.gen[rindex] == NOINDEX and random() < ratio:
          self.gen[rindex] = ACTIVE
        elif self.gen[rindex] == ACTIVE and random() < 1 - ratio:
            self.gen[rindex] = 0

    @property
    def scalarize(self):
        return sum([
            (x * y) / z for x, y, z in zip(self._fitness, self.fitness_weights, FITNESS_LIMTIS)
        ])


class GenixSingleGen(RandomTableGen):

    @property
    def fitness(self):
        return self.scalarize

    def is_dominant(self, other):
        return self.fitness < other.fitness


class GenixMultiGen(RandomTableGen):

    @property
    def fitness(self):
        return self._fitness


class Deme:
    def __init__(
            self,
            name,
            columns,
            flat_defs,
            from_pos,
            is_multiobjective=False,
            fitness_weights=[],
        ):
        self.name = name
        self.len_columns = len(columns)
        self.from_pos = from_pos
        GenixElem = GenixMultiGen if is_multiobjective else GenixSingleGen
        self.population = [
            GenixElem(
                index_defs=flat_defs,
                initial_fitness=(float('inf'), float('inf')),
                ratio=0.2,
                from_pos=from_pos,
                length=self.len_columns,
                fitness_weights=fitness_weights,
            ) for _ in columns
        ]
        self.immigrants = []
        # change according the Algorithm type (i.e. if it is pareto)
        self.is_multiobjective = is_multiobjective
        self._metadata = {
            "offline": [],
            "online": [],
        }

    def immigrate(self, source_deme, only_elites=False, ratio=0.2):
        for elem in source_deme.population:
            bool_op = all if only_elites else any
            if bool_op([random() < ratio, elem.is_elite]):
                self.immigrants.append(Immigrant(deepcopy(elem), source_deme.name))

class SA:
    def __init__(self, total_iterations, temp):
        self.temp = temp
        self.total_iterations = total_iterations
        self.ratio = temp / total_iterations

    def reduce(self):
        val =  1 - math.exp(-self.temp/self.total_iterations)
        self.temp -= self.ratio
        return val


class Genix:
    def __init__(
            self,
            is_multiobjective=False,
            run_parallel=True,
            benchmark_queries=[],
            migration_policy=BIDIRECTIONAL,
            db_settings={},
            fitness_weights=[],
        ):
        self.demes = {}
        self.db_settings = db_settings
        self.benchmark_queries = benchmark_queries
        self.is_multiobjective = is_multiobjective
        self.run_parallel = run_parallel
        self.migration_policy = migration_policy
        self.fitness_weights = fitness_weights
        self._create_demes()
        self._cache = LocalCache(benchmark_queries)

    def _create_demes(self):
        pos = 0
        map_defs = get_defs(self.db_settings.get("db_src"))
        self.flat_defs = []
        for name, columns in map_defs.items():
            for column in columns:
                self.flat_defs.append((name, column))
        for k, v in map_defs.items():
            deme = Deme(
                k, v, self.flat_defs, pos,
                is_multiobjective=self.is_multiobjective,
                fitness_weights=self.fitness_weights,
            )
            self.demes[k] = deme
            pos += len(v)

    def print_population(self):
        for deme in self.demes.values():
            print("deme: ", deme.name)
            for i in deme.population:
                print(i.get_gen_str())

    def _migrate(self):
        for deme in self.demes.values():
            deme.immigrants = []
        for name_from, connected_demes in edges.items():
            for name_to in connected_demes:
                deme_a = self.demes[name_from]
                deme_b = self.demes[name_to]
                if self.migration_policy in [LEFT_TO_RIGHT, BIDIRECTIONAL]:
                    deme_a.immigrate(deme_b, only_elites=False, ratio=0.05)
                if self.migration_policy in [RIGHT_TO_LEFT, BIDIRECTIONAL]:
                    deme_b.immigrate(deme_a, only_elites=False, ratio=0.05)

    def _plot_results(self, results, writer, iter):
        if self.use_pareto:
            self._plot_results_pareto(results, writer, iter)
        else:
            self._plot_results_aggregation(results, writer, iter)

    def evolve(self, num_generations):
        sa = SA(num_generations, 0)
        str_weights = "__".join([str(x) for x in self.fitness_weights])
        str_mode = "multi" if self.is_multiobjective else "single"
        logdir = get_log_dir(
            "temp_runs/" if is_debug else "runs/",
            f"{str_mode}__{str_weights}")
        writer = SummaryWriter(logdir=logdir)
        cpu_count = mp.cpu_count()
        runner_options = self.db_settings
        runner_options["benchmark_queries"] = self.benchmark_queries
        runner_options["cache"] = self._cache
        for iter in range(num_generations):
            metropolis = sa.reduce()
            new_population = {}
            if self.run_parallel:
                pool = mp.Pool(cpu_count)
                def callback(result):
                    key, value = result
                    new_population[key] = value
                for deme in self.demes.values():
                    pool.apply_async(
                        evolve_deme,
                        args=(
                            deme,
                            self.is_multiobjective,
                            run_index,
                            metropolis,
                            runner_options,
                        ),
                        callback=callback,
                        error_callback=error_callback,
                    )
                pool.close()
                pool.join()
            else:
                for deme in self.demes.values():
                    key, value = evolve_deme(
                        deme,
                        self.is_multiobjective,
                        run_index,
                        metropolis,
                        runner_options,
                    )
                    new_population[key] = value

            # Copy new population
            for deme, population in new_population.items():
                self.demes[deme].population = population

            self._migrate()
            plot_generation(
                writer,
                self.demes,
                iter,
                flat_defs=self.flat_defs,
                is_multiobjective=self.is_multiobjective,
            )


def error_callback(exception):
    print("(mp)", exception)


if __name__ == "__main__":

    args = parser.parse_args()
    is_debug = args.debug

    num_generations = 1 if is_debug else 3
    benchmark_queries = BENCHMARK_QUERIES
    if is_debug:
        benchmark_queries = TEST_QUERIES
    genix = Genix(
        is_multiobjective=False,
        benchmark_queries=benchmark_queries,
        db_settings={
            "db_src": "dbs/TPC-H-small.db",
            "db_dest": "./replications" if is_debug else "/dev/shm",
        },
        fitness_weights=[0.7, 0.3],
        run_parallel=not is_debug,
    )
    genix.evolve(num_generations=num_generations)
