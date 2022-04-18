from copy import copy
import os, errno
from collections import defaultdict
from random import randint, random, shuffle
import numpy as np
import queries
import math
import shutil
import sqlite3
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


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


class BaseGen:
    def __init__(self, index_defs):
        self.process_id = str(uuid.uuid4())[:8]
        self.index_defs = index_defs
        self.gen = [0 for _ in index_defs]
        self.is_elite = False

    def prepare_queries(self):
        indexes = [v for k, v in zip(self.gen, self.index_defs) if k]
        table_columns = defaultdict(list)
        for k, v in indexes:
            table_columns[k].append(v)
        queries = []
        for table, columns in table_columns.items():
            str_columns = ", ".join(columns)
            index_name = "__".join(columns)
            index_name = f"{table}__{index_name}"
            queries.append(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({str_columns});"
            )

        return queries

    def mutate(self):
        rindex = randint(0, len(self.gen) - 1)
        self.gen[rindex] = 0 if self.gen[rindex] else 1

    def crossover(self, other):
        rindex = randint(0, len(self.gen) - 2)
        self.gen[:rindex], other.gen[:rindex] = other.gen[:rindex], self.gen[:rindex]

    def leveling(self, ratio=0.5):
        for i in range(len(self.gen)):
            if self.gen[i] == 1 and random() < 0.5:
                self.gen[i] = 0



class RandomGen(BaseGen):
    def __init__(self, index_defs, ratio=0.5):
        super().__init__(index_defs)
        self.gen = [randint(0, 1) * randint(0, 1) for _ in index_defs]
        # for i, _ in enumerate(index_defs):
        #     if random() > ratio:
        #         self.gen[i] = 1


class PositionGen(BaseGen):
    def __init__(self, index_defs, index=0):
        super().__init__(index_defs)
        lindex = index if len(index_defs) > index else 0
        self.gen[lindex] = 1


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


def prepare_query_index(index_def):
    column_names = ', '.join(index_def.columns)
    query = f"CREATE INDEX IF NOT EXISTS {index_def.name} ON {index_def.table}({column_names});"
    return query


def run_index(db_name, elem):
    mem_dbname = copy_db(db_name, elem.process_id)
    mem_db = sqlite3.connect(mem_dbname)
    cursor = mem_db.cursor()
    queries = elem.prepare_queries()
    for query in queries:
        cursor.execute(query)

    total_time = 0

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
    return (elem.process_id, total_time)


def get_top_indexes(population, n=1):
    # population
    i_population = list(enumerate(population.copy()))
    i_population.sort(key=lambda a: a[1][1])

    return [k for k, _ in i_population][:n]


def main():
    db_src = "dbs/TPC-H-small.db"
    defs = get_defs(db_src)
    cpu_count = mp.cpu_count()
    # It is important to have the number of population based in the number of cpu
    # in order to have proportional execution times.
    len_population = int(math.ceil(len(defs) / cpu_count) * cpu_count)
    population = [PositionGen(defs, i) for i in range(len_population)]
    # population = [RandomGen(defs, 0.2) for i in range(len_population)]
    epochs = 30

    for i in range(epochs):

        profiles = []
        def callback(r):
            profiles.append(r)
        pool = mp.Pool(6)
        for elem in population:
            pool.apply_async(run_index, args=(db_src, elem), callback=callback)
        pool.close()
        pool.join()

        print(min(profiles, key=lambda x: x[1]))

        indexes = get_top_indexes(profiles, n=3)
        epoch_population = [copy(population[i]) for i in indexes]
        for elem in epoch_population:
            elem.is_elite = True

        while len(epoch_population) < len_population:
            rindex = randint(0, len_population - 1)
            new_elem = copy(population[rindex])
            new_elem.is_elite = False
            epoch_population.append(new_elem)

        shuffle(epoch_population)

        for elem in epoch_population:
            if elem.is_elite:
                continue
            if random() < 0.7:
                elem.mutate()
            if random() < 0.2:
                rindex = randint(0, len_population - 1)
                elem.crossover(epoch_population[rindex])
            if random() < 0.3:
                elem.leveling()

        population = epoch_population

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
    main()
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
