from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from ga_helper import set_elitism_multi_objective, set_elitism_single_objective


def __(text):
    if text is None or text == "":
        return ""
    return f"```{text}```"


def get_population_fig(population, elites, draw_pareto=False):
    fig, ax = plt.subplots()
    for item in population:
        x, y = item._fitness
        ax.scatter(x, y, label=item.process_id, alpha=1 if item.is_elite else 0.2)
    ax.legend()
    if draw_pareto:
        felites = [x._fitness for x in elites]
        felites.sort(key=lambda x: x[0])
        lx = np.array([x[0] for x in felites])
        ly = np.array([x[1] for x in felites])
        ax.plot(lx, ly)
    # plt.show()
    return fig


def test_plot(points):
    fig, ax = plt.subplots()
    x = [a[0] for a in points]
    y = [a[1] for a in points]
    # ax.scatter(x, y, c=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=.2)
    ax.scatter([0, 2], [0, 1], label="Elites: 1234567, 123445")
    # ax.plot(x, y)
    ax.legend()
    # for x in points:
    #     ax.scatter(x[0], x[1])
    print(tikzplotlib.get_tikz_code())
    plt.show()


def _write_queries(writer, map_demes, name, iter, flat=False):
    data_values = {
        "deme": [],
        "pid": [],
        "gen": [],
        "time": [],
        "space": [],
        "score": [],
        "index_query": [],
        "elite": [],
    }
    for deme_name, deme in map_demes.items():
        population = []
        if flat:
            population = deme
        else:
            population = deme.population
        for elem in population:
            data_values["pid"].append(elem.process_id)
            data_values["deme"].append(deme_name)
            data_values["gen"].append(__(elem.get_gen_str()))
            data_values["score"].append(round(elem.scalarize, 4))
            data_values["time"].append(round(elem._fitness[0], 4))
            data_values["space"].append(round(elem._fitness[1], 4))
            data_values["index_query"].append(__("".join(elem.get_phenotype_queries())))
            data_values["elite"].append(__("true" if elem.is_elite else "false"))

    df = pd.DataFrame(data=data_values)
    df = df.set_index('pid')
    writer.add_text(f"queries/{name}", df.to_markdown(), iter)


def _write_scalar(writer, deme, iter, mode="offline"):
    delta = 1 / (iter + 1)
    avg_value = np.average(
        [
            x.scalarize
            for x in deme.population
            if mode == "offline" and x.is_elite or mode == "online"
        ]
    )
    deme._metadata[mode].append(avg_value)
    hist_value = np.sum(deme._metadata[mode]) * delta
    writer.add_scalar(f"{mode}/{deme.name}", hist_value, iter)
    if mode == "offline":
        writer.add_scalar(f"best-so-far/{deme.name}", avg_value, iter)
    return hist_value


def _write_fitness_points(writer, population, name, iter, is_multiobjective):
    elites = [elem for elem in population if elem.is_elite]
    # fig = get_population_fig(
    #     population,
    #     elites,
    #     draw_pareto=is_multiobjective,
    # )
    # writer.add_figure(f"non-dominants/{name}", fig, iter)

    return elites


def _write_all_best_so_far(writer, population, iter):
    avg_value = np.average(
        [
            elem.scalarize for elem in population if elem.is_elite
        ]
    )
    writer.add_scalar(f"best-so-far/all", avg_value, iter)


def _write_histogram(writer, columns, histogram, iter):
    data_values = {
        "column": columns,
        "value": histogram,
    }
    df = pd.DataFrame(data=data_values)
    df = df.set_index('column')
    writer.add_text(f"best-elem/histogram", df.to_markdown(), iter)


def get_histogram(population):
    return [sum(i) for i in zip(*[elem.gen for elem in population])]


def plot_generation(writer, demes, iter, flat_defs, is_multiobjective):
    all_online = 0
    all_offline = 0
    # all_non_dominants = {}
    elites = []
    map_elites = {}
    for deme in demes.values():
        deme_elites = []
        online = _write_scalar(writer, deme, iter, mode="online")
        offline = _write_scalar(writer, deme, iter, mode="offline")
        all_online += online
        all_offline += offline
        non_dominants = _write_fitness_points(
            writer, deme.population, deme.name, iter, is_multiobjective
        )
        # all_non_dominants[deme.name] = non_dominants
        for elem in non_dominants:
            new_elem = deepcopy(elem)
            deme_elites.append(new_elem)
        elites += deme_elites
        map_elites[deme.name] = deme_elites
    len_demes = len(demes.values())
    _write_queries(writer, demes, "all", iter)
    writer.add_scalar(f"online/all", all_online / len_demes, iter)
    writer.add_scalar(f"offline/all", all_offline / len_demes, iter)
    if is_multiobjective:
        set_elitism_multi_objective(elites)
    else:
        set_elitism_single_objective(elites)
    _write_all_best_so_far(writer, elites, iter)
    _write_histogram(writer, flat_defs, get_histogram(elites), iter)
    _write_fitness_points(
        writer, elites, "all", iter, is_multiobjective=is_multiobjective
    )
    _write_queries(writer, map_elites, "elites", iter, flat=True)
