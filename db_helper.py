import os, errno
import shutil
import sqlite3
import queries
import time
from local_cache import get_hash_key


BENCHMARK_QUERIES = [
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
    ("query 15", queries.query_15),
    ("query 16", queries.query_16),
    ("query 17", queries.query_17),
    ("query 18", queries.query_18),
    ("query 19", queries.query_19),
    ("query 20", queries.query_20),
    ("query 21", queries.query_21),
]

TEST_QUERIES = [
    ("query 1", queries.query_1),
]


def get_defs(db_src, plain=True):
    file_db = sqlite3.connect(db_src)
    query_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    cursor = file_db.execute(query_all_tables)
    table_names = [row[0] for row in cursor.fetchall()]
    result_dict = {}
    for table_name in table_names:
        query_columns = f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');"
        cursor = file_db.execute(query_columns)
        result_dict[table_name] = [row[0] for row in cursor.fetchall()]
    file_db.close()

    return result_dict


def copy_db(src, dest_path, dest_name):
    dest_path = os.path.join(dest_path, dest_name)
    os.mkdir(dest_path)
    dest_file = os.path.join(dest_path, 'main.db')
    shutil.copyfile(src, dest_file)

    return dest_path, dest_file


def silentremove(filedir):
    try:
        shutil.rmtree(filedir)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred


def get_index_size_query(phenotype):
    names = ", ".join([f"'{x[0]}'" for x in phenotype])
    query = f"select COALESCE(sum(pgsize), 0) from 'dbstat' where name IN ({names});"
    return query

def _run_index(elem, db_src, db_dest, benchmark_queries):
    db_path, db_file = copy_db(db_src, db_dest, elem.process_id)
    mem_db = sqlite3.connect(db_file)
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
    for _, query in benchmark_queries:
        s = time.perf_counter()
        cursor.executescript(query)
        cursor.fetchone()
        e = time.perf_counter()
        query_time = e - s
        total_time += query_time
    mem_db.close()
    silentremove(db_path)
    return (total_time, query_size)


def run_index(elem, db_src, db_dest, benchmark_queries, cache):
    key = get_hash_key(elem.gen)
    value = cache.get(key)
    if value is None:
        value = _run_index(elem, db_src, db_dest, benchmark_queries)
    else:
        value = tuple(value)
    return value
