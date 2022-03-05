import queries
import sqlite3
import time
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


def run_index(db_name, process_number):
    file_db = sqlite3.connect(db_name)
    # mem_db = sqlite3.connect('/mnt/tmp/tmp1.db')
    # query = "".join(line for line in file_db.iterdump())
    # mem_db.executescript(query)
    cursor = file_db.cursor()
    total_time = 0
    for query_number, query in all_queries:
        s = time.time()
        cursor.execute(query)
        cur_result = cursor.fetchone()
        e = time.time()
        local_time = e - s
        total_time += local_time
        print(process_number, "query:", query_number, "took:", local_time)
    print("Process number:", process_number, "took", total_time, "s")


def main():
    pool = mp.Pool(2)
    pool.starmap(run_index, [("TPC-H-small.db", "p1", ), ("TPC-H-small.db", "p2", )])

if __name__ == '__main__':
    s = time.time()
    main()
    # p1 = Process(target=run_index, args=("TPC-H-small.db", "p1"))
    # p1.start()
    # p2 = Process(target=run_index, args=("TPC-H-small.db", "p2"))
    # p2.start()
    # p1.join()
    # p2.join()

    # run_index("TPC-H-small.db", "p")
    # run_index("TPC-H-small.db", "p2")


    e = time.time()
    print("took: ", e - s)
    # Process(target=run_index, args=("TPC-H-small.db", "p"))
    # Process(target=run_index, args=("TPC-H-small.db", "p2"))
    # run_index("TPC-H-small.db")
