import os, errno
import shutil
import sqlite3
import queries


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


def get_defs(db_name, plain=True):
    file_db = sqlite3.connect(db_name)
    query_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    cursor = file_db.execute(query_all_tables)
    table_names = [row[0] for row in cursor.fetchall()]
    # result = []
    result_dict = {}
    for table_name in table_names:
        query_columns = f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');"
        cursor = file_db.execute(query_columns)
        result_dict[table_name] = [row[0] for row in cursor.fetchall()]
        # result += [(table_name, row[0]) for row in cursor.fetchall()]
    file_db.close()

    return result_dict


def copy_db(src, dst):
    path_dst = f'/dev/shm/db_{dst}.db'
    # path_dst = f'./replications/db_{dst}.db'
    shutil.copyfile(src, path_dst)

    return path_dst


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred