from copy import copy, deepcopy
from collections import defaultdict
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

