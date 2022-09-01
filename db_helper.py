import os, errno
import shutil
import sqlite3


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
    shutil.copyfile(src, path_dst)

    return path_dst


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred