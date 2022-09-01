from copy import copy, deepcopy
import sqlite3
import os, errno
from collections import defaultdict
from random import randint, random, shuffle
import numpy as np
import queries
import math
import shutil
import time
import uuid
import multiprocessing as mp


all_queries = [
    ("query 1", queries.query_1),
    ("query 2", queries.query_2),
    ("query 3", queries.query_3),
    ("query 4", queries.query_4),
    ("query 5", queries.query_5),
    ("query 6", queries.query_6),
    ("query 7", queries.query_7),
    ("query 8", queries.query_8),
    ("query 9", queries.query_9),
    ("query 10", queries.query_10),
    ("query 11", queries.query_11),
    ("query 12", queries.query_12),
    ("query 13", queries.query_13),
    ("query 14", queries.query_14),
    # ("query 15", queries.query_15),
    ("query 16", queries.query_16),
    # ("query 17", queries.query_17),
    ("query 18", queries.query_18),
    ("query 19", queries.query_19),
    # ("query 20", queries.query_20),
    ("query 21", queries.query_21),
]

# SELECT name FROM sqlite_master WHERE type='table';
# SELECT name FROM PRAGMA_TABLE_INFO('NATION');
# Get query index size
# select name, sum(pgsize) as size from dbstat  WHERE name = "idx_lineitem_ltax" group by name;


# def get_tables(connection):

MAX_TOTAL_TIME = 10
MAX_INDEX_SIZE = 100_000_000


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


ACTIVE = 1
NOINDEX = 0


class BaseGen:
    def __init__(self, index_defs):
        self.assign_process_id()
        self.index_defs = index_defs
        self.gen = [0 for _ in index_defs]
        self.is_elite = False

    def get_gen_str(self):
        array_str = ["█" if i else "·" for i in self.gen]
        return "|" + "".join(array_str) + "|"

    @classmethod
    def init_from_gen(cls, index_def, gen):
        instance = cls(index_def)
        instance.gen = copy(gen)
        return instance

    def assign_process_id(self):
        self.process_id = str(uuid.uuid4())[:8]

    def get_phenotype(self):
        indexes = [v for k, v in zip(self.gen, self.index_defs) if k]

        print("--> indexes", indexes)

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

    def shuffle_gen(self):
        shuffle(self.gen)

    def leveling(self, ratio=0.5):
        for i in range(len(self.gen)):
            if self.gen[i] == ACTIVE and random() < ratio:
                self.gen[i] = NOINDEX

    def pick_best(self, other, ratio=0.5):
        for i in range(len(self.gen)):
            if other.gen[i] == ACTIVE and random() < ratio:
                self.gen[i] = other.gen[i]


class RandomGen(BaseGen):
    def __init__(self, index_defs, ratio=0.5):
        super().__init__(index_defs)
        self.gen = [ACTIVE if random() < ratio else NOINDEX for _ in index_defs]
        # for i, _ in enumerate(index_defs):
        #     if random() > ratio:
        #         self.gen[i] = 1


class PositionGen(BaseGen):
    def __init__(self, index_defs, index=0):
        super().__init__(index_defs)
        lindex = index if len(index_defs) > index else 0
        self.gen[lindex] = ACTIVE


def get_defs(db_name):
    file_db = sqlite3.connect(db_name)
    query_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    cursor = file_db.execute(query_all_tables)
    table_names = [row[0] for row in cursor.fetchall()]
    result = []
    for table_name in table_names:
        query_columns = f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');"
        cursor = file_db.execute(query_columns)
        result += [(table_name, row[0]) for row in cursor.fetchall()]
    file_db.close()

    return result


def copy_db(src, dst):
    path_dst = f'/dev/shm/db_{dst}.db'
    shutil.copyfile(src, path_dst)

    return path_dst


def get_index_size_query(phenotype):
    names = ", ".join([f"'{x[0]}'" for x in phenotype])
    query = f"select COALESCE(sum(pgsize), 0) from 'dbstat' where name IN ({names});"
    return query


def run_index(db_name, elem):
    mem_dbname = copy_db(db_name, elem.process_id)
    mem_db = sqlite3.connect(mem_dbname)
    cursor = mem_db.cursor()
    phenotype = elem.get_phenotype()
    queries = elem.get_phenotype_queries(phenotype)

    for query in queries:
        cursor.execute(query)

    total_time = 0

    index_size_query_query = get_index_size_query(phenotype)

    pointer = cursor.execute(index_size_query_query)
    qq = pointer.fetchone()
    query_size = qq[0]

    for query_number, query in all_queries:
        s = time.perf_counter()
        cursor.execute(query)
        cursor.fetchone()
        e = time.perf_counter()
        local_time = e - s
        total_time += local_time
        # print(f"\t{elem.process_id},{local_time}")
    mem_db.close()
    silentremove(mem_dbname)
    # print(f"{elem.process_id},{total_time},{len(queries)}")
    return (elem.process_id, total_time, query_size)


def normalize_rewards(rewards):
    return [
        (
            i[0],
            ((i[1] * 0.85) / MAX_TOTAL_TIME) + ((i[2] * 0.15) / MAX_INDEX_SIZE),
        ) for i in rewards
    ]


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


def get_top_n(rewards, n=1):
    return sorted(rewards, key=lambda a: a[1])[:n]


def single_thread():
    db_src = "dbs/TPC-H-small.db"
    defs = get_defs(db_src)
    test_gens = [
        RandomGen(defs, 0.075)
        for i in range(10)
    ]
    rewards = []
    for g in test_gens:
        rewards.append(run_index(db_src, g))
    get_top_n(rewards, 2)


def test():
    db_src = "dbs/TPC-H-small.db"
    defs = get_defs(db_src)
    cpu_count = mp.cpu_count()
    # It is important to have the number of population based in the number of cpus
    # in order to have proportional execution times.
    len_defs = len(defs)
    len_population = int(math.ceil(len_defs / cpu_count) * cpu_count)
    # population = [PositionGen(defs, i % len_defs) for i in range(len_population)]
    # population = [RandomGen(defs, 0.075) for _ in range(len_population)]

    test_gens = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]

    population = [
        RandomGen.init_from_gen(defs, item) for item in test_gens
    ]

    # for i, elem in enumerate(population):
    #     print(elem.process_id, i, elem.is_elite)
    #     print(elem.gen)
    #     print("\n".join(elem.get_phenotype_queries()))
    #     print("\n")

    all_elites = []

    epochs = 1

    for i in range(epochs):
        len_population = len(population)
        # print("*** iteration:", i + 1)

        profiles = []
        def callback(r):
            profiles.append(r)
        pool = mp.Pool(6)
        for elem in population:
            pool.apply_async(run_index, args=(db_src, elem), callback=callback)
            # profiles.append(run_index(db_src, elem))

        pool.close()
        pool.join()
        n = math.ceil(len_population * 0.075)

        # print("profiles", profiles)
        top_profiles = get_top_n(profiles, n=n)

        # print(profiles)
        # print(top_profiles[0])
        top_profile_1 = top_profiles[0]
        map_population = {elem.process_id: elem for elem in population}

        for k, v in profiles:
            print(map_population[k].get_gen_str(), v)

        # print("indexes", indexes)
        epoch_population = [copy(map_population[process_id]) for process_id, _ in top_profiles]

        for elem in epoch_population:
            elem.is_elite = True

        best_elem = map_population[top_profile_1[0]]

        all_elites.append(best_elem)

        # print(best_elem.get_gen_str(), i, top_profile_1)

        while len(epoch_population) < len_population:
            rindex = randint(0, len_population - 1)
            new_elem = copy(population[rindex])
            new_elem.assign_process_id()
            new_elem.is_elite = False
            epoch_population.append(new_elem)

        shuffle(epoch_population)

        # for i, elem in enumerate(epoch_population):
        #     print(elem.process_id, i, elem.is_elite)
        #     print(elem.gen)
        #     print("\n".join(elem.get_phenotype_queries()))
        #     print("\n")


        for elem in epoch_population:
            if elem.is_elite:
                rindex = randint(0, len_population - 1)
                if not epoch_population[rindex].is_elite:
                    epoch_population[rindex].pick_best(elem)
            else:
                # if random() < 0.7:
                if random() < 1:
                    elem.mutate()
                if random() < 1:
                # if random() < 0.2:
                    rindex = randint(0, len_population - 1)
                    elem.crossover(
                        epoch_population[rindex],
                        partial=epoch_population[rindex].is_elite
                    )
                # if random() < 0.3:
                #     elem.leveling()

        # print("############# afteeeeer rouletting\n")

        # for i, elem in enumerate(epoch_population):
        #     print(elem.process_id, i, elem.is_elite)
        #     print(elem.gen)
        #     print("\n".join(elem.get_phenotype_queries()))
        #     print("\n")

        population = epoch_population


def main(use_pareto=False):
    db_src = "dbs/TPC-H-small.db"
    defs = get_defs(db_src)
    cpu_count = mp.cpu_count()
    # It is important to have the number of population based in the number of cpus
    # in order to have proportional execution times.
    len_defs = len(defs)
    # len_population = int(math.ceil(len_defs / cpu_count) * cpu_count)
    len_population = 54
    # population = [PositionGen(defs, i % len_defs) for i in range(len_population)]
    population = [RandomGen(defs, 0.075) for _ in range(len_population)]
    means = []

    all_elites = []

    epochs = 50

    for i in range(epochs):
        len_population = len(population)

        raw_rewards = []
        def callback(r):
            raw_rewards.append(r)
        pool = mp.Pool(6)
        for elem in population:
            pool.apply_async(run_index, args=(db_src, elem), callback=callback)

        pool.close()
        pool.join()

        if use_pareto:
            top_rewards = get_pareto(raw_rewards)
        else:
            rewards = normalize_rewards(raw_rewards)
            n_prof = math.ceil(len_population * 0.075)
            top_rewards = get_top_n(rewards, n=n_prof)

        top_profile_1 = top_rewards[0]
        map_population = {elem.process_id: elem for elem in population}

        # for k, v in rewards:
        #     print(map_population[k].get_gen_str(), v)

        epoch_population = [deepcopy(map_population[reward[0]]) for reward in top_rewards]

        for elem in epoch_population:
            elem.is_elite = True

        best_elem = map_population[top_profile_1[0]]

        all_elites.append(deepcopy(best_elem))
        if use_pareto:
            # for k in top_rewards:
            plain_top_rewards = {i[0]: i for i in top_rewards}
            for reward in raw_rewards:
                x = map_population[reward[0]]
                print(x.get_gen_str(), x.process_id, "%.3f" % reward[1], "%.3f" % reward[2], "*" if x.process_id in plain_top_rewards else " ")
            obj1 = [x[1] for x in raw_rewards]
            obj2 = [x[2] for x in raw_rewards]
            means.append([np.mean(obj1), np.mean(obj2)])
        else:
            for k, v in rewards:
                x = map_population[k]
                print(x.get_gen_str(), x.process_id, "%.3f" % v, "*" if best_elem.process_id == x.process_id else " ")
            print("best elem", best_elem.get_gen_str(), i, top_profile_1)

        print("\n")



        while len(epoch_population) < len_population:
            rindex = randint(0, len_population - 1)
            new_elem = copy(population[rindex])
            new_elem.assign_process_id()
            new_elem.is_elite = False
            epoch_population.append(new_elem)

        shuffle(epoch_population)

        # for i, elem in enumerate(epoch_population):
        #     print(elem.process_id, i, elem.is_elite)
        #     print(elem.gen)
        #     print("\n".join(elem.get_phenotype_queries()))
        #     print("\n")


        for elem in epoch_population:
            if elem.is_elite:
                rindex = randint(0, len_population - 1)
                if not epoch_population[rindex].is_elite:
                    epoch_population[rindex].pick_best(elem)
            else:
                if random() < 0.7:
                    elem.mutate()
                if random() < 0.2:
                    rindex = randint(0, len_population - 1)
                    elem.crossover(
                        epoch_population[rindex],
                        partial=epoch_population[rindex].is_elite
                    )
                if random() < 0.05:
                    elem.shuffle_gen()
                # if random() < 0.3:
                #     elem.leveling()

        # print("############# afteeeeer rouletting\n")

        # for i, elem in enumerate(epoch_population):
        #     print(elem.process_id, i, elem.is_elite)
        #     print(elem.gen)
        #     print("\n".join(elem.get_phenotype_queries()))
        #     print("\n")
        population = epoch_population

    print("means", means)

    # for i, elem in enumerate(all_elites):
    #     print(elem.process_id, i, elem.is_elite)
    #     print(elem.gen)
    #     print("\n".join(elem.get_phenotype_queries()))
    #     print("\n")

    # print("\n", "final", "\n")

    # for i, elem in enumerate(population):
    #     print(elem.process_id, i, elem.is_elite)
    #     print(elem.gen)
    #     print("\n".join(elem.get_phenotype_queries()))
    #     print("\n")

    # last_profiles = []
    # def callback(r):
    #     last_profiles.append(r)
    # pool = mp.Pool(6)
    # for elem in population:
    #     pool.apply_async(run_index, args=(db_src, elem), callback=callback)
    # pool.close()
    # pool.join()

    # print(np.mean([prof[1] for prof in profiles]))

    # for elem in population:
    #     print(elem.gen)



    # print(run_index(db_src, population[0]))


    # print(results)
    # print("-->", get_top_indexes(results))


if __name__ == '__main__':
    main(use_pareto=True)
    # single_thread()
    # p1 = Process(target=run_index, args=("TPC-H-small.db", "p1"))
    # p1.start()
    # p2 = Process(target=run_index, args=("TPC-H-small.db", "p2"))
    # p2.start()
    # p1.join()
    # p2.join()

    # run_index("TPC-H-small.db", "p")
    # run_index("TPC-H-small.db", "p2")


    # e = time.time()
    # print("took: ", e - s)
    # Process(target=run_index, args=("TPC-H-small.db", "p"))
    # Process(target=run_index, args=("TPC-H-small.db", "p2"))
    # run_index("TPC-H-small.db")





# def main():
#     db_src = "dbs/TPC-H-small.db"
#     defs = get_defs(db_src)
#     cpu_count = mp.cpu_count()
#     # It is important to have the number of population based in the number of cpus
#     # in order to have proportional execution times.
#     len_defs = len(defs)
#     len_population = int(math.ceil(len_defs / cpu_count) * cpu_count)
#     # population = [PositionGen(defs, i % len_defs) for i in range(len_population)]
#     population = [RandomGen(defs, 0.075) for _ in range(len_population)]

#     for elem in population:
#         print("\n".join(elem.get_phenotype_queries()), "\n")

#     epochs = 30

#     # for i in range(epochs):

#     #     profiles = []
#     #     def callback(r):
#     #         profiles.append(r)
#     #     pool = mp.Pool(6)
#     #     for elem in population:
#     #         pool.apply_async(run_index, args=(db_src, elem), callback=callback)
#     #     pool.close()
#     #     pool.join()

#     #     print(min(profiles, key=lambda x: x[1]))

#     #     indexes = get_top_indexes(profiles, n=3)
#     #     epoch_population = [copy(population[i]) for i in indexes]
#     #     for elem in epoch_population:
#     #         elem.is_elite = True

#     #     while len(epoch_population) < len_population:
#     #         rindex = randint(0, len_population - 1)
#     #         new_elem = copy(population[rindex])
#     #         new_elem.is_elite = False
#     #         epoch_population.append(new_elem)

#     #     shuffle(epoch_population)

#     #     for elem in epoch_population:
#     #         if elem.is_elite:
#     #             continue
#     #         if random() < 0.7:
#     #             elem.mutate()
#     #         if random() < 0.2:
#     #             rindex = randint(0, len_population - 1)
#     #             elem.crossover(epoch_population[rindex])
#     #         if random() < 0.3:
#     #             elem.leveling()

#     #     population = epoch_population

#     # last_profiles = []
#     # def callback(r):
#     #     last_profiles.append(r)
#     # pool = mp.Pool(6)
#     # for elem in population:
#     #     pool.apply_async(run_index, args=(db_src, elem), callback=callback)
#     # pool.close()
#     # pool.join()

#     # print(np.mean([prof[1] for prof in profiles]))

#     # for elem in population:
#     #     print(elem.gen)



#     # print(run_index(db_src, population[0]))


#     # print(results)
#     # print("-->", get_top_indexes(results))



# def main():
#     db_src = "dbs/TPC-H-small.db"
#     defs = get_defs(db_src)
#     cpu_count = mp.cpu_count()
#     # It is important to have the number of population based in the number of cpus
#     # in order to have proportional execution times.
#     len_defs = len(defs)
#     len_population = int(math.ceil(len_defs / cpu_count) * cpu_count)
#     # population = [PositionGen(defs, i % len_defs) for i in range(len_population)]
#     # population = [RandomGen(defs, 0.075) for _ in range(len_population)]

#     gen1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#     gen2 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

#     elem1 = RandomGen.init_from_gen(defs, gen1)
#     elem2 = RandomGen.init_from_gen(defs, gen2)
#     population = [elem1, elem2]

#     for i, elem in enumerate(population):
#         print(elem.process_id, i, elem.is_elite)
#         print(elem.gen)
#         print("\n".join(elem.get_phenotype_queries()))
#         print("\n")

#     epochs = 1

#     for i in range(epochs):

#         profiles = []
#     #     def callback(r):
#     #         profiles.append(r)
#     #     pool = mp.Pool(6)
#         for elem in population:
#             profiles.append(run_index(db_src, elem))

#         print(min(profiles, key=lambda x: x[1]))

#         indexes = get_top_n(profiles, n=3)

#         print(profiles)

#         print("indexes", indexes)
#         # epoch_population = [copy(population[i]) for i in indexes]


#         # for elem in epoch_population:
#         #     elem.is_elite = True

#     #     while len(epoch_population) < len_population:
#     #         rindex = randint(0, len_population - 1)
#     #         new_elem = copy(population[rindex])
#     #         new_elem.assign_process_id()
#     #         new_elem.is_elite = False
#     #         epoch_population.append(new_elem)

#     #     shuffle(epoch_population)

#     #     for elem in epoch_population:
#     #         if elem.is_elite:
#     #             continue
#     #         if random() < 0.7:
#     #             elem.mutate()
#     #         if random() < 0.2:
#     #             rindex = randint(0, len_population - 1)
#     #             elem.crossover(epoch_population[rindex])
#     #         if random() < 0.3:
#     #             elem.leveling()

#     #     population = epoch_population

#     # last_profiles = []
#     # def callback(r):
#     #     last_profiles.append(r)
#     # pool = mp.Pool(6)
#     # for elem in population:
#     #     pool.apply_async(run_index, args=(db_src, elem), callback=callback)
#     # pool.close()
#     # pool.join()

#     # print(np.mean([prof[1] for prof in profiles]))

#     # for elem in population:
#     #     print(elem.gen)



#     # print(run_index(db_src, population[0]))


#     # print(results)
#     # print("-->", get_top_indexes(results))
