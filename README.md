# Evoindexing

A database indexing by using evolutionary algorithms.


## Install

### Create Database Benchmark
- Clone the repository: https://github.com/lovasoa/TPCH-sqlite
- Generate a 0.1 version of TPC-H DB with: `SCALE_FACTOR=0.1 make`

### Install dependencies
- Install rocksDB version 7

### Instal the project

```pip install -r requirements.txt```


## Model

### Genotype

Array of 0s and 1s where each position represents if the index is active or not

### Phenotype


```
CREATE INDEX IF NOT EXISTS PARTSUPP__PS_SUPPLYCOST ON PARTSUPP(PS_SUPPLYCOST);
CREATE INDEX IF NOT EXISTS CUSTOMER__C_NAME ON CUSTOMER(C_NAME);
CREATE INDEX IF NOT EXISTS ORDERS__O_CUSTKEY__O_ORDERSTATUS ON ORDERS(O_CUSTKEY, O_ORDERSTATUS);
```

### Query

```
select * from "dbstat" where name like "index_name";
```

The returned size is in _bytes_


Another value we can use.

```
select sum(pgsize-unused)*100.0/sum(pgsize) from "dbstat" where name like "index_name";
```

To see how efficiently the content of a table is stored on disk, compute the amount of space used to hold actual content divided by the total amount of disk space used. The closer this number is to 100%, the more efficient the packing. (In this example, the 'xyz' table is assumed to be in the 'main' schema. Again, there are two different versions that show the use of DBSTAT both without and with the new aggregated feature, respectively.)


### Results

After 50 iterations



```
38931f0e 49 True
[0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1]
CREATE INDEX IF NOT EXISTS NATION__N_NAME__N_COMMENT ON NATION(N_NAME, N_COMMENT);
CREATE INDEX IF NOT EXISTS PART__P_MFGR__P_SIZE ON PART(P_MFGR, P_SIZE);
CREATE INDEX IF NOT EXISTS SUPPLIER__S_NATIONKEY ON SUPPLIER(S_NATIONKEY);
CREATE INDEX IF NOT EXISTS PARTSUPP__PS_SUPPKEY__PS_SUPPLYCOST ON PARTSUPP(PS_SUPPKEY, PS_SUPPLYCOST);
CREATE INDEX IF NOT EXISTS CUSTOMER__C_CUSTKEY__C_PHONE ON CUSTOMER(C_CUSTKEY, C_PHONE);
CREATE INDEX IF NOT EXISTS ORDERS__O_ORDERDATE__O_ORDERPRIORITY__O_CLERK__O_SHIPPRIORITY ON ORDERS(O_ORDERDATE, O_ORDERPRIORITY, O_CLERK, O_SHIPPRIORITY);
CREATE INDEX IF NOT EXISTS LINEITEM__L_SUPPKEY__L_LINENUMBER__L_DISCOUNT__L_TAX__L_RETURNFLAG__L_SHIPDATE__L_COMMENT ON LINEITEM(L_SUPPKEY, L_LINENUMBER, L_DISCOUNT, L_TAX, L_RETURNFLAG, L_SHIPDATE, L_COMMENT);

|·█·█·····█··█······█····█·█·█···█·······████···██··███·█····█|





65ac7b0b 49 False
[0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]
CREATE INDEX IF NOT EXISTS NATION__N_NAME ON NATION(N_NAME);
CREATE INDEX IF NOT EXISTS REGION__R_NAME ON REGION(R_NAME);
CREATE INDEX IF NOT EXISTS PART__P_PARTKEY__P_TYPE__P_RETAILPRICE ON PART(P_PARTKEY, P_TYPE, P_RETAILPRICE);
CREATE INDEX IF NOT EXISTS SUPPLIER__S_NAME__S_NATIONKEY__S_ACCTBAL__S_COMMENT ON SUPPLIER(S_NAME, S_NATIONKEY, S_ACCTBAL, S_COMMENT);
CREATE INDEX IF NOT EXISTS PARTSUPP__PS_SUPPKEY__PS_SUPPLYCOST__PS_COMMENT ON PARTSUPP(PS_SUPPKEY, PS_SUPPLYCOST, PS_COMMENT);
CREATE INDEX IF NOT EXISTS CUSTOMER__C_ACCTBAL ON CUSTOMER(C_ACCTBAL);
CREATE INDEX IF NOT EXISTS ORDERS__O_CUSTKEY__O_ORDERPRIORITY__O_COMMENT ON ORDERS(O_CUSTKEY, O_ORDERPRIORITY, O_COMMENT);
CREATE INDEX IF NOT EXISTS LINEITEM__L_SUPPKEY__L_TAX__L_RETURNFLAG__L_LINESTATUS__L_COMMITDATE__L_COMMENT ON LINEITEM(L_SUPPKEY, L_TAX, L_RETURNFLAG, L_LINESTATUS, L_COMMITDATE, L_COMMENT);

|·█···█·█···█··█··█·█·██·█·██·····█···█···█··█··█····███·█···█|





86856082 49 False
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
CREATE INDEX IF NOT EXISTS NATION__N_NAME ON NATION(N_NAME);
CREATE INDEX IF NOT EXISTS PART__P_CONTAINER__P_RETAILPRICE ON PART(P_CONTAINER, P_RETAILPRICE);
CREATE INDEX IF NOT EXISTS SUPPLIER__S_NATIONKEY__S_PHONE ON SUPPLIER(S_NATIONKEY, S_PHONE);
CREATE INDEX IF NOT EXISTS PARTSUPP__PS_SUPPKEY__PS_AVAILQTY__PS_SUPPLYCOST ON PARTSUPP(PS_SUPPKEY, PS_AVAILQTY, PS_SUPPLYCOST);
CREATE INDEX IF NOT EXISTS CUSTOMER__C_NATIONKEY__C_PHONE ON CUSTOMER(C_NATIONKEY, C_PHONE);
CREATE INDEX IF NOT EXISTS ORDERS__O_CUSTKEY__O_ORDERDATE__O_COMMENT ON ORDERS(O_CUSTKEY, O_ORDERDATE, O_COMMENT);
CREATE INDEX IF NOT EXISTS LINEITEM__L_SUPPKEY__L_TAX__L_SHIPDATE ON LINEITEM(L_SUPPKEY, L_TAX, L_SHIPDATE);

|·█···········██····██···███····██····█··█···█··█····█··█·····|

```
