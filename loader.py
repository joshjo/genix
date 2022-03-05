import sqlite3
from io import StringIO


sqlite_database = 'TPC-H-small.db'


query_1 = """
select
	l_returnflag,
	l_linestatus,
	sum(l_quantity) as sum_qty,
	sum(l_extendedprice) as sum_base_price,
	sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
	sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
	avg(l_quantity) as avg_qty,
	avg(l_extendedprice) as avg_price,
	avg(l_discount) as avg_disc,
	count(*) as count_order
from
	lineitem
where
	l_shipdate <= date '1998-12-01' - interval ':1' day
group by
	l_returnflag,
	l_linestatus
order by
	l_returnflag,
	l_linestatus
LIMIT 1;
"""


query_3 = """
SELECT
    l_orderkey,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    o_orderdate,
    o_shippriority
FROM
    customer,
    orders,
    lineitem
WHERE
    c_mktsegment = 'BUILDING'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < '1995-03-15'
    AND l_shipdate > '1995-03-15'
GROUP BY
    l_orderkey,
    o_orderdate,
    o_shippriority
ORDER BY
    revenue desc,
    o_orderdate
LIMIT 20;
"""


query_a = "SELECT COUNT(*) FROM customer;"



def init_sqlite_db():
    # Read database to tempfile
    con = sqlite3.connect(sqlite_database)
    tempfile = StringIO()
    for line in con.iterdump():
        tempfile.write('%s\n' % line)
    con.close()
    tempfile.seek(0)

    # Create a database in memory and import from tempfile
    sqlite = sqlite3.connect(":memory:")
    sqlite.cursor().executescript(tempfile.read())
    sqlite.commit()
    # sqlite.row_factory = sqlite3.Row
    cursor = sqlite.cursor()
    cursor.execute(query_a)
    cur_result = cursor.fetchone()
    print(cur_result)
    sqlite.close()


# if __name__ == 'main':
init_sqlite_db()
