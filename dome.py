from collections import defaultdict
from copy import copy, deepcopy
from random import random, randint, shuffle
import math
import multiprocessing as mp
import sqlite3
import time
import uuid

from db_helper import (
    BENCHMARK_QUERIES,
    copy_db,
    get_defs,
    silentremove,
)
from ga_helper import get_top_n, normalize_rewards

ACTIVE = 1
NOINDEX = 0


# The relation is based is on who received the connection
edges = {
    'NATION': ['REGION'],
    'SUPPLIER': ['NATION'],
    'PARTSUPP': ['PART', 'SUPPLIER'],
    'CUSTOMER': ['NATION'],
    'ORDERS': ['CUSTOMER'],
    'LINEITEM': ['ORDERS', 'PARTSUPP'],
}


def clean_column_name(column):
    return column.split("_", 1)[1]


def get_map_pk(defs):
    result = {}
    for k, v in defs.items():
        key = clean_column_name(v[0])
        result[key] = k
    return result


def get_edges(defs):
    map_pk = get_map_pk(defs)
    result = defaultdict(list)
    for k, v in defs.items():
        for i in range(1, len(v)):
            column_name = v[i]
            if column_name.endswith("KEY"):
                key = clean_column_name(column_name)
                table_name = map_pk.get(key)
                if table_name:
                    result[k].append(table_name)
    return result


def get_index_size_query(phenotype):
    names = ", ".join([f"'{x[0]}'" for x in phenotype])
    query = f"select COALESCE(sum(pgsize), 0) from 'dbstat' where name IN ({names});"
    return query


def run_index(db_name, elem):
    mem_dbname = copy_db(db_name, elem.process_id)
    mem_db = sqlite3.connect(mem_dbname)
    cursor = mem_db.cursor()
    phenotype = elem.get_phenotype()
    create_index_queries = elem.get_phenotype_queries(phenotype)

    for query in create_index_queries:
        cursor.execute(query)

    total_time = 0

    index_size_query_query = get_index_size_query(phenotype)
    pointer = cursor.execute(index_size_query_query)
    qq = pointer.fetchone()
    query_size = qq[0]

    for _, query in BENCHMARK_QUERIES:
        s = time.perf_counter()
        cursor.executescript(query)
        cursor.fetchone()
        e = time.perf_counter()
        query_time = e - s
        total_time += query_time
    mem_db.close()
    silentremove(mem_dbname)
    return (elem.process_id, round(total_time, 1), query_size)



class BaseGen:
    def __init__(self, index_defs):
        self.gen = [NOINDEX for _ in index_defs]
        # print("index_defs", index_defs)
        self.index_defs = index_defs
        self.assign_process_id()

    def pick_best(self, other, ratio=0.5):
        for i in range(len(self.gen)):
            if other.gen[i] == ACTIVE and random() < ratio:
                self.gen[i] = other.gen[i]

    def mutate(self, ratio=0.2):
        """
        The chance of removing an index should be greater than adding
        """
        rindex = randint(0, len(self.gen) - 1)
        if self.gen[rindex] == NOINDEX and random() < ratio:
          self.gen[rindex] = ACTIVE
        elif self.gen[rindex] == ACTIVE and random() < 1 - ratio:
            self.gen[rindex] = 0

    def crossover(self, other, partial=False):
        rindex = randint(0, len(self.gen) - 2)
        if partial:
            self.gen[:rindex] = other.gen[:rindex]
        else:
            self.gen[:rindex], other.gen[:rindex] = other.gen[:rindex], self.gen[:rindex]

    def get_gen_str(self):
        array_str = ["█" if i else "·" for i in self.gen]
        return "|" + "".join(array_str) + "|"

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
    def __init__(self, index_defs, ratio=0.5, from_pos=0, length=0):
        super().__init__(index_defs)
        size = len(index_defs)
        to_pos = min(from_pos + length, size)
        for i in range(from_pos, to_pos):
            self.gen[i] = ACTIVE if random() < ratio else NOINDEX
        self.from_pos = from_pos
        self.to_pos = to_pos

    def mutate(self, ratio=0.2):
        """
        The chance of removing an index should be greater than adding
        """
        rindex = randint(self.from_pos, self.to_pos)
        if self.gen[rindex] == NOINDEX and random() < ratio:
          self.gen[rindex] = ACTIVE
        elif self.gen[rindex] == ACTIVE and random() < 1 - ratio:
            self.gen[rindex] = 0

class Dome:
    def __init__(self, name, columns, flat_defs, from_pos):
        self.name = name
        self.len_columns = len(columns)
        self.from_pos = from_pos
        self.population = [
            RandomTableGen(flat_defs, ratio=0.2, from_pos=from_pos, length=self.len_columns)
            for _ in columns
        ]

    def evaluate(self, db_src):
        raw_rewards = []
        for individual in self.population:
            raw_rewards.append(run_index(db_name=db_src, elem=individual))
        normalized_rewards = normalize_rewards(raw_rewards)
        return raw_rewards, normalized_rewards

    def evolve(self, db_src, metropolis=0):
        """
        :param progress: a value from 0 to 1 which determines the progress of the iterations
            if progress is 0 then it is starting. if it is 1 then all
        """
        len_population = len(self.population)
        n_prof = math.ceil(len_population * 0.075)
        raw_rewards, rewards = self.evaluate(db_src)

        top_rewards = get_top_n(rewards, n=n_prof)
        map_population = {elem.process_id: elem for elem in self.population}
        epoch_population = [map_population[reward[0]] for reward in top_rewards]

        for elem in epoch_population:
            elem.is_elite = True

        for elem, reward in zip(self.population, raw_rewards):
            print(elem.process_id, elem.get_gen_str(), reward)

        shuffle(epoch_population)

        while len(epoch_population) < len_population:
            rindex = randint(0, len_population - 1)
            new_elem = copy(self.population[rindex])
            new_elem.assign_process_id()
            new_elem.is_elite = False
            epoch_population.append(new_elem)

        for elem in epoch_population:
            if elem.is_elite and not random() < metropolis:
                continue
            if random() < 0.3:
                elem.mutate()
            if random() < 0.7:
                rindex = randint(0, len_population - 1)
                elem.crossover(
                    epoch_population[rindex],
                    partial=epoch_population[rindex].is_elite,
                )

        # for elem in epoch_population:
        #     print(elem.process_id, elem.get_gen_str())
                # if random() < 0.05:
                #     elem.shuffle_gen()
        self.population = epoch_population

    def random_interchange(self, other):
        self_rand_index = randint(0, len(self.population) - 1)
        other_rand_index = randint(0, len(other.population) - 1)
        print("self index", self_rand_index, "to index", other_rand_index)
        temp = self.population[self_rand_index]
        self.population[self_rand_index] = other.population[other_rand_index]
        other.population[other_rand_index] = temp


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
    def __init__(self, num_evolutions=2):
        self.domes = []
        self.map_domes = {}
        self.db_src = "dbs/TPC-H-small.db"
        self.num_evolutions = num_evolutions
        init_temp = 0
        self.sa = SA(num_evolutions, init_temp)

    def create_domes(self):
        pos = 0
        map_defs = get_defs(self.db_src)
        flat_defs = []
        for name, columns in map_defs.items():
            for column in columns:
                flat_defs.append((name, column))
        for k, v in map_defs.items():
            dome = Dome(k, v, flat_defs, pos)
            self.domes.append(dome)
            self.map_domes[k] = dome
            pos += len(v)


    def print_population(self):
        for index, dome in enumerate(self.domes):
            print("dome: ", dome.name)
            for i in dome.population:
                print(i.get_gen_str())

    def _emigrate(self):
        for name_from, connected_domes in edges.items():
            for name_to in connected_domes:
                if random() < 0.15:
                    dome_from = self.map_domes[name_from]
                    dome_to = self.map_domes[name_to]
                    dome_from.random_interchange(dome_to)

    def evolve(self):
        for i in range(self.num_evolutions):
            metropolis = self.sa.reduce()
            # pool = mp.Pool(6)
            # for dome in self.domes:
            #     pool.apply_async(dome.evolve, args=(self.db_src, metropolis), )
            # pool.close()
            # pool.join()
            for dome in self.domes:
                dome.evolve(self.db_src, metropolis)
            self._emigrate()
            self.print_population()
                # dome.evolve(self.db_src, metropolis)


def dump(seconds):
    print("waiting", seconds, "seconds")
    time.sleep(seconds)
    print("finish", seconds, "seconds")


class Parallel:
    def __init__(self, domes):
        self.domes = domes

    def evolve(self):
        for iter in range(1):
            # pool = mp.Pool(6)
            for dome in self.domes:
                dump(dome)
                # pool.apply_async(dump, args=(dome,))
            # pool.close()
            # pool.join()



if __name__ == "__main__":
    # p = Parallel([2, 4, 3, 7, 4])
    # p.evolve()
    genix = Genix()
    genix.create_domes()
    genix.evolve()
    # genix.print_population()
    # genix.print_population()








    # print("-->", genix.map_domes)
    # # genix.print_population()
    # genix.evolve()
    # total_iterations = 100
    # sa = SA(15, total_iterations)

    # for i in range(total_iterations):
    #     print(sa.reduce())
