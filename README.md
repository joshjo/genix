# Evoindexing

Get best indexes by using eovlutionary algorithms.


## Install

- Clone the repository: https://github.com/lovasoa/TPCH-sqlite
- Generate a 0.1 version of TPC-H DB with: `SCALE_FACTOR=0.1 make`


## TODO:

- Get all column from all the tables as an bitarray:

This means that position 1 and 6 are turn on:

[0, 1, 0, 0, 0, 0, 1].

This will produce a list of single o composite index queries.

- Get the size of the active indexes.

