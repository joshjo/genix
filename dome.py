from collections import defaultdict
from random import random
from db_helper import get_defs


ACTIVE = 1
NOINDEX = 0

# The relation is based is on who received the connection
edges = {
    'NATION': ['REGION'],
    'SUPPLIER': ['NATION'],
    'PARTSUPP': ['PART', 'SUPPLIER'],
    'CUSTOMER': ['NATION'],
    'ORDERS': ['CUSTOMER'],
    'LINEITEM': ['ORDERS', 'PARTSUPP'],
}


def clean_column_name(column):
    return column.split("_", 1)[1]


def get_map_pk(defs):
    result = {}
    for k, v in defs.items():
        key = clean_column_name(v[0])
        result[key] = k
    return result


def get_edges(defs):
    map_pk = get_map_pk(defs)
    result = defaultdict(list)
    for k, v in defs.items():
        for i in range(1, len(v)):
            column_name = v[i]
            if column_name.endswith("KEY"):
                key = clean_column_name(column_name)
                table_name = map_pk.get(key)
                if table_name:
                    result[k].append(table_name)
    return result
        # result[k]


class BaseGen:
    def __init__(self, size):
        self.gen = [NOINDEX for _ in range(size)]

    def get_gen_str(self):
        array_str = ["█" if i else "·" for i in self.gen]
        return "|" + "".join(array_str) + "|"


class RandomTableGen(BaseGen):
    def __init__(self, size, ratio=0.5, from_pos=0, length=0):
        super().__init__(size)
        to_pos = min(from_pos + length, size)
        for i in range(from_pos, to_pos):
            self.gen[i] = ACTIVE if random() < ratio else NOINDEX


class Dome:
    def __init__(self, name, columns, size, from_pos):
        self.name = name
        len_columns = len(columns)
        self.population = [
            RandomTableGen(size, ratio=0.2, from_pos=from_pos, length=len_columns)
            for _ in columns
        ]


class Genix:
    def __init__(self):
        self.domes = []

    # def migrate(self):
    #     for


if __name__ == "__main__":
    db_src = "dbs/TPC-H-small.db"
    defs = get_defs(db_src)

    # print("defs", defs)

    print(dict(get_edges(defs)))
    # print(get_map_pk(defs))


    # total_columns = sum([len(v) for x, v in defs.items()])

    # domes = []
    # pos = 0
    # for k, v in defs.items():
    #     domes.append(Dome(k, v, total_columns, pos))
    #     pos += len(v)

    # for index, dome in enumerate(domes):
    #     print("dome: ", index + 1)
    #     for i in dome.population:
    #         print(i.get_gen_str())



    # print("defs", defs)
    # xxx = ["a" for i in range(60)]
    # gen = RandomTableGen(xxx, ratio=1, from_pos=2, length=100)

    # print("-->", gen.gen)
