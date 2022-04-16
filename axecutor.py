import queries
import shutil
import sqlite3
import time
import uuid
import multiprocessing as mp
from dataclasses import dataclass


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

class Index:
    def __init__(self, table, columns):
        self.process_id = str(uuid.uuid4())[:8]
        self.table = table
        self.columns = columns

    @property
    def name(self):
        column_names = '__'.join(self.columns)
        return f"idx_{self.table}__{column_names}".lower()

    def __repr__(self) -> str:
        return self.name


def create_indexes(table_name, columns):
    return [Index(table_name, [column]) for column in columns]


def get_defs(db_name):
    file_db = sqlite3.connect(db_name)
    query_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    cursor = file_db.execute(query_all_tables)
    table_names = [row[0] for row in cursor.fetchall()]
    result = []
    for table_name in table_names:
        query_columns = f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');"
        cursor = file_db.execute(query_columns)
        cols = [row[0] for row in cursor.fetchall()]
        result += create_indexes(table_name, cols)

    file_db.close()

    return result


def get_mem_db_name(index_def):
    return f'/dev/shm/db_{index_def.process_id}.db'


def copy_db(src, index_def):
    shutil.copyfile(src, get_mem_db_name(index_def))


def prepare_query_index(index_def):
    column_names = ', '.join(index_def.columns)
    query = f"CREATE INDEX IF NOT EXISTS {index_def.name} ON {index_def.table}({column_names});"
    return query

def run_index(db_name, index_def):
    copy_db(db_name, index_def)
    mem_dbname = get_mem_db_name(index_def)
    mem_db = sqlite3.connect(mem_dbname)
    cursor = mem_db.cursor()
    # query_index = prepare_query_index(index_def)
    # cursor.execute(query_index)

    total_time = 0

    for query_number, query in all_queries:
        s = time.time()
        cursor.execute(query)
        cursor.fetchone()
        e = time.time()
        local_time = e - s
        total_time += local_time
        print(f"{query_number},{local_time}")
    print(f"{index_def.process_id},{index_def.name},{total_time}")
    mem_db.close()


def main():
    defs = get_defs("TPC-H-small.db")
    # run_index("TPC-H-small.db", defs[10])
    # pool = mp.Pool(2)
    # pool.starmap(run_index, [("TPC-H-small.db", "p1", ), ("TPC-H-small.db", "p2", )])

    pool = mp.Pool()
    for index_def in defs[-1:]:
        pool.apply_async(run_index, args=("TPC-H-small.db", index_def))
    pool.close()
    pool.join()


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
