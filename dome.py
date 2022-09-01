from collections import defaultdict
from random import random
import multiprocessing as mp
import sqlite3
import time
import uuid

from db_helper import copy_db, get_defs, silentremove
import queries

ACTIVE = 1
NOINDEX = 0

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
    mem_db.close()
    silentremove(mem_dbname)
    # print(f"{elem.process_id},{total_time},{len(queries)}")
    return (elem.process_id, total_time, query_size)



class BaseGen:
    def __init__(self, index_defs):
        self.gen = [NOINDEX for _ in index_defs]
        # print("index_defs", index_defs)
        self.index_defs = index_defs
        self.assign_process_id()

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


class Dome:
    def __init__(self, name, columns, flat_defs, from_pos):
        self.name = name
        len_columns = len(columns)
        self.population = [
            RandomTableGen(flat_defs, ratio=0.2, from_pos=from_pos, length=len_columns)
            for _ in columns
        ]


class Genix:
    def __init__(self):
        self.domes = []
        self.db_src = "dbs/TPC-H-small.db"

    def create_domes(self):
        pos = 0
        map_defs = get_defs(self.db_src)
        flat_defs = []
        for name, columns in map_defs.items():
            for column in columns:
                flat_defs.append((name, column))
        for k, v in map_defs.items():
            self.domes.append(Dome(k, v, flat_defs, pos))
            pos += len(v)

    def print_population(self):
        for index, dome in enumerate(self.domes):
            print("dome: ", index + 1)
            for i in dome.population:
                print(i.get_gen_str())

    def evolve(self):
        for dome in self.domes:
            print("dome: ", dome.name)
            for individual in dome.population:
                result = run_index(db_name=self.db_src, elem=individual)
                print("--> result", result)
            print("----")


if __name__ == "__main__":
    genix = Genix()
    genix.create_domes()
    genix.print_population()
    genix.evolve()
