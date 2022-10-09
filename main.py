from collections import defaultdict
from copy import copy, deepcopy
from random import random, randint, shuffle, uniform
from dotenv import dotenv_values
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import sqlite3
from tensorboardX import SummaryWriter
import time
import uuid

from db_helper import (
    BENCHMARK_QUERIES,
    copy_db,
    get_defs,
    silentremove,
)
from ga_helper import get_top_n, normalize_rewards, get_pareto

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

def plot_pareto():
    N = 21
    x = np.linspace(0, 10, 11)
    y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1,  9.9, 13.9, 15.1, 12.5]

    # fit a linear curve an estimate its y-values and their error.
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1/len(x) +
                            (x - x.mean())**2 / np.sum((x - x.mean())**2))

    fig, ax = plt.subplots()
    ax.plot(x, y_est, '-')
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
    ax.plot(x, y, 'o', color='tab:brown')

    return fig

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


def get_index_size_query(phenotype):
    names = ", ".join([f"'{x[0]}'" for x in phenotype])
    query = f"select COALESCE(sum(pgsize), 0) from 'dbstat' where name IN ({names});"
    return query


def run_index(db_name, elem):
    mem_dbname = copy_db(db_name, elem.process_id)
    mem_db = sqlite3.connect(mem_dbname)
    cursor = mem_db.cursor()
    phenotype = elem.get_phenotype()
    create_index_queries = elem.get_phenotype_queries(phenotype)

    for query in create_index_queries:
        cursor.execute(query)

    total_time = 0

    index_size_query_query = get_index_size_query(phenotype)
    pointer = cursor.execute(index_size_query_query)
    qq = pointer.fetchone()
    query_size = qq[0]

    for _, query in BENCHMARK_QUERIES:
        s = time.perf_counter()
        cursor.executescript(query)
        cursor.fetchone()
        e = time.perf_counter()
        query_time = e - s
        total_time += query_time
    mem_db.close()
    silentremove(mem_dbname)
    return (elem.process_id, round(total_time, 1), query_size)



class BaseGen:
    def __init__(self, index_defs):
        self.gen = [NOINDEX for _ in index_defs]
        # print("index_defs", index_defs)
        self.index_defs = index_defs
        self.assign_process_id()

    def pick_best(self, other, ratio=0.5):
        for i in range(len(self.gen)):
            if other.gen[i] == ACTIVE and random() < ratio:
                self.gen[i] = other.gen[i]

    def mutate(self, ratio=0.2):
        """
        The chance of removing an index should be greater than adding
        """
        rindex = randint(0, len(self.gen) - 1)
        if self.gen[rindex] == NOINDEX and random() < ratio:
          self.gen[rindex] = ACTIVE
        elif self.gen[rindex] == ACTIVE and random() < 1 - ratio:
            self.gen[rindex] = 0

    def crossover(self, other, partial=False):
        rindex = randint(0, len(self.gen) - 2)
        if partial:
            self.gen[:rindex] = other.gen[:rindex]
        else:
            self.gen[:rindex], other.gen[:rindex] = other.gen[:rindex], self.gen[:rindex]

    def get_gen_str(self):
        array_str = ["█" if i else "·" for i in self.gen]
        return "|" + "".join(array_str) + "|"

    def assign_process_id(self):
        self.process_id = str(uuid.uuid4())[:8]

    def get_phenotype(self):
        indexes = [v for k, v in zip(self.gen, self.index_defs) if k]
        table_columns = defaultdict(list)
        for k, v in indexes:
            table_columns[k].append(v)
        triplets = []
        for table, columns in table_columns.items():
            str_columns = ", ".join(columns)
            index_name = "__".join(columns)
            index_name = f"{table}__{index_name}"
            triplets.append((index_name, table, str_columns))

        return triplets

    def get_phenotype_queries(self, phenotype=None):
        if phenotype is None:
            phenotype = self.get_phenotype()
        return [
            f"CREATE INDEX IF NOT EXISTS {indexname} ON {table}({columns});"
            for indexname, table, columns in phenotype
        ]


class RandomTableGen(BaseGen):
    def __init__(self, index_defs, ratio=0.5, from_pos=0, length=0):
        super().__init__(index_defs)
        size = len(index_defs)
        to_pos = min(from_pos + length, size)
        for i in range(from_pos, to_pos):
            self.gen[i] = ACTIVE if random() < ratio else NOINDEX
        self.from_pos = from_pos
        self.to_pos = to_pos

    def mutate(self, ratio=0.2):
        """
        The chance of removing an index should be greater than adding
        """
        rindex = randint(self.from_pos, self.to_pos - 1)
        if self.gen[rindex] == NOINDEX and random() < ratio:
          self.gen[rindex] = ACTIVE
        elif self.gen[rindex] == ACTIVE and random() < 1 - ratio:
            self.gen[rindex] = 0


class Elem:
    def __init__(self, value):
        self.individual = [value for _ in range(10)]
        self.process_id = str(uuid.uuid4())[:8]

    def __repr__(self):
        return str(self.individual)

    def update_value(self, value):
        for i in range(len(self.individual)):
            self.individual[i] += value


class Deme:
    def __init__(self, name, columns, flat_defs, from_pos, use_pareto=False):
        # self.population = [Elem(2) for _ in columns]
        self.name = name
        self.len_columns = len(columns)
        self.from_pos = from_pos
        self.population = [
            RandomTableGen(flat_defs, ratio=0.2, from_pos=from_pos, length=self.len_columns)
            for _ in columns
        ]
        # change according the Algorithm type (i.e. if it is pareto)
        self.use_pareto = use_pareto
        self._metadata = {
            "averages": [],
            "bests": [],
        }

    def evaluate(self, db_src):
        raw_rewards = []
        for individual in self.population:
            raw_rewards.append(run_index(db_name=db_src, elem=individual))
        normalized_rewards = normalize_rewards(raw_rewards)
        return raw_rewards, normalized_rewards

    def evolve(self, db_src, metropolis):
        """
        :param progress: a value from 0 to 1 which determines the progress of the iterations
            if progress is 0 then it is starting. if it is 1 then all
        """
        len_population = len(self.population)
        n_top = math.ceil(len_population * 0.075)
        raw_rewards, rewards = self.evaluate(db_src)
        epoch_population = []
        map_population = {elem.process_id: elem for elem in self.population}
        summary = {}

        if self.use_pareto:
            selected_rewards = get_pareto(raw_rewards)
            summary["non_dominants"] = selected_rewards
        else:
            sorted_rewards = sorted(rewards, key=lambda a: a[1])
            selected_rewards = sorted_rewards[:n_top]
            summary["sorted_rewards"] = sorted_rewards
        epoch_population = [map_population[reward[0]] for reward in selected_rewards]

        summary["new_population"] = epoch_population

        for elem in epoch_population:
            elem.is_elite = True

        shuffle(epoch_population)

        while len(epoch_population) < len_population:
            rindex = randint(0, len_population - 1)
            new_elem = copy(self.population[rindex])
            new_elem.assign_process_id()
            new_elem.is_elite = False
            epoch_population.append(new_elem)

        for elem in epoch_population:
            if elem.is_elite and not random() < metropolis:
                continue
            if random() < 0.3:
                elem.mutate()
            if random() < 0.7:
                rindex = randint(0, len_population - 1)
                elem.crossover(
                    epoch_population[rindex],
                    partial=epoch_population[rindex].is_elite,
                )
        return self.name, summary

    def random_interchange(self, other):
        self_rand_index = randint(0, len(self.population) - 1)
        other_rand_index = randint(0, len(other.population) - 1)
        temp = self.population[self_rand_index]
        self.population[self_rand_index] = other.population[other_rand_index]
        other.population[other_rand_index] = temp


class SA:
    def __init__(self, total_iterations, temp):
        self.temp = temp
        self.total_iterations = total_iterations
        self.ratio = temp / total_iterations

    def reduce(self):
        val =  1 - math.exp(-self.temp/self.total_iterations)
        self.temp -= self.ratio
        return val


class Genix:
    def __init__(self, num_generations=0):
        self.demes = []
        self.map_demes = {}
        self.db_src = "dbs/TPC-H-small.db"
        self.num_generations = num_generations
        init_temp = 0
        self.sa = SA(num_generations, init_temp)

    def create_demes(self):
        pos = 0
        map_defs = get_defs(self.db_src)
        flat_defs = []
        for name, columns in map_defs.items():
            for column in columns:
                flat_defs.append((name, column))
        for k, v in map_defs.items():
            deme = Deme(k, v, flat_defs, pos)
            self.demes.append(deme)
            self.map_demes[k] = deme
            pos += len(v)


    def print_population(self):
        for index, deme in enumerate(self.demes):
            print("deme: ", deme.name)
            for i in deme.population:
                print(i.get_gen_str())

    def _emigrate(self):
        num_emigrations = 0
        for name_from, connected_demes in edges.items():
            for name_to in connected_demes:
                if random() < 0.4:
                    deme_from = self.map_demes[name_from]
                    deme_to = self.map_demes[name_to]
                    deme_from.random_interchange(deme_to)
                    num_emigrations += 1
        return num_emigrations

    def merge_results(self, results, writer, iter):
        all_offline = 0
        all_online = 0
        all_best_elems = []
        for deme in self.demes:
            result = results[deme.name]
            new_population = result["new_population"]
            sorted_rewards = result["sorted_rewards"]
            best_so_far_id, best_so_far_score = sorted_rewards[0]
            avg_value = np.average([x[1] for x in sorted_rewards])
            deme._metadata["bests"].append(best_so_far_score)
            deme._metadata["averages"].append(avg_value)

            deme.population = new_population
            delta = 1 / (iter + 1);
            offline = np.sum(deme._metadata["bests"]) * delta
            online = np.sum(deme._metadata["averages"]) * delta


            for best_item in new_population:
                if best_item.process_id == best_so_far_id:
                    break

            writer.add_scalar(f"best-so-far/{deme.name}", best_so_far_score, iter)
            writer.add_scalar(f"offline/{deme.name}", offline, iter)
            writer.add_scalar(f"online/{deme.name}", online, iter)

            # best_item = deme.population[0]
            all_queries = "\n".join(best_item.get_phenotype_queries())
            elem_string = f"gen: \n {best_item.get_gen_str()} \n\n pheno: \n {all_queries}"
            writer.add_text(f"best-elem/{deme.name}", elem_string, iter)

            # Summary
            all_offline += offline
            all_online += online
            all_best_elems.append([best_so_far_score, best_item, deme.name])
            # Use this for pareto
            # fig = plot_pareto()
        len_demes = len(self.demes)
        all_best_elems.sort(key=lambda x: x[0])
        best_elem_score, best_item, best_elem_deme_name = all_best_elems[0]
        elem_string = f"gen: \n {best_item.get_gen_str()} \n\n pheno: \n {all_queries} \n\n deme: {best_elem_deme_name}"
        writer.add_text(f"best-elem/all", elem_string, iter)
        writer.add_scalar(f"best-so-far/all", best_elem_score, iter)
        writer.add_scalar(f"offline/all", all_offline / len_demes, iter)
        writer.add_scalar(f"online/all", online / len_demes, iter)

    def evolve(self):
        writer = SummaryWriter()
        cpu_count = mp.cpu_count()
        for iter in range(self.num_generations):
            # print("iter:", iter)
            metropolis = self.sa.reduce()
            results = {}
            pool = mp.Pool(cpu_count)
            def callback(result):
                key, value = result
                results[key] = value
            for deme in self.demes:
                pool.apply_async(deme.evolve, args=(self.db_src, metropolis), callback=callback)
            pool.close()
            pool.join()

            self.merge_results(results, writer, iter)
            num_migrations = self._emigrate()

            writer.add_scalar(f"summary/emigrations", num_migrations, iter)


def dump(seconds):
    print("waiting", seconds, "seconds")
    time.sleep(seconds)
    print("finish", seconds, "seconds")


if __name__ == "__main__":
    values = dotenv_values(".env")
    # p = Parallel([2, 4, 3, 7, 4])
    # p.evolve()
    # env =
    genix = Genix(num_generations=50)
    genix.create_demes()
    genix.evolve()
    # genix.print_population()
    # genix.print_population()



    # print("-->", genix.map_domes)
    # # genix.print_population()
    # genix.evolve()
    # total_iterations = 100
    # sa = SA(15, total_iterations)

    # for i in range(total_iterations):
    #     print(sa.reduce())
