import sqlite3
import queries
import time


queries = [
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
    ("query 17", queries.query_17),
    ("query 18", queries.query_18),
    ("query 19", queries.query_19),
    ("query 20", queries.query_20),
    ("query 21", queries.query_21),
]


# class DummyQueries:
#     queries = [
#         queries.query_a,
#     ]


# class RunTPCH(TPCQueries):

#     def __init__(self, db_name):
#         self.db_name = db_name
#         self.connection = sqlite3.connect(self.db_name)

#     def execute(self):
#         cursor = self.connection.cursor()
#         # index_a = "CREATE INDEX IF NOT EXISTS index_a ON lineitem(l_partkey);"
#         # index_b = "CREATE INDEX IF NOT EXISTS index_b ON supplier(s_suppkey);"
#         index_a = "DROP INDEX IF EXISTS index_a;"
#         index_b = "DROP INDEX IF EXISTS index_b;"
#         cursor.execute(index_a)
#         cursor.execute(index_b)
#         for query_number, query in self.queries:
#             s = time.time()
#             cursor.execute(query)
#             cur_result = cursor.fetchone()
#             e = time.time()
#             print("query:", query_number, "took:", e - s)
#         self.connection.close()
#         # return cur_result


def run_index(db_name, index_query):
    file_db = sqlite3.connect(db_name)
    # mem_db = sqlite3.connect('/mnt/tmp/tmp1.db')
    # query = "".join(line for line in file_db.iterdump())
    # mem_db.executescript(query)
    cursor = file_db.cursor()
    for query_number, query in enumerate(queries):
        s = time.time()
        cursor.execute(query)
        cur_result = cursor.fetchone()
        e = time.time()
        print("query:", query_number, "took:", e - s)


if __name__ == '__main__':
    run_index('TPC-H-small.db', 'tmp')

    # runner = RunTPCH('TPC-H-small.db')
    # Get index size.
    # SELECT SUM("pgsize") FROM "dbstat" WHERE name='index_a';
    # runner.execute()


# def init_sqlite_db():
#     # Read database to tempfile
#     con = sqlite3.connect(sqlite_database)
#     tempfile = StringIO()
#     for line in con.iterdump():
#         tempfile.write('%s\n' % line)
#     con.close()
#     tempfile.seek(0)

#     # Create a database in memory and import from tempfile


"""
(indexing) ➜  Indexing python executor.py
----> 0.04672741889953613
(indexing) ➜  Indexing python executor.py
----> 19.032994985580444
(indexing) ➜  Indexing
(indexing) ➜  Indexing python executor.py
(indexing) ➜  Indexing python executor.py
----> 0.08546900749206543
(indexing) ➜  Indexing python executor.py
----> 18.467249155044556
(indexing) ➜  Indexing python executor.py
----> 338.52814412117004

"""
