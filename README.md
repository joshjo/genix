# Evoindexing

A database indexing by using eovlutionary algorithms.


## Install

- Clone the repository: https://github.com/lovasoa/TPCH-sqlite
- Generate a 0.1 version of TPC-H DB with: `SCALE_FACTOR=0.1 make`


## Model

### Genotype

Array of 0s and 1s where each position represents if the index is active or not

### Phenotype


```
CREATE INDEX IF NOT EXISTS PARTSUPP__PS_SUPPLYCOST ON PARTSUPP(PS_SUPPLYCOST);
CREATE INDEX IF NOT EXISTS CUSTOMER__C_NAME ON CUSTOMER(C_NAME);
CREATE INDEX IF NOT EXISTS ORDERS__O_CUSTKEY__O_ORDERSTATUS ON ORDERS(O_CUSTKEY, O_ORDERSTATUS);
```
