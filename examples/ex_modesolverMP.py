"""Fully vectorial finite-difference mode solver example."""

import numpy as np
import EMpy
import pylab
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt


def epsfunc(x_, y_):
    """Return a matrix describing a 2d material.

    :param x_: x values
    :param y_: y values
    :return: 2d-matrix
    """
    xx, yy = np.meshgrid(x_, y_)
    return np.where(
        (np.abs(xx.T - 1.24e-6) <= 0.24e-6) * (np.abs(yy.T - 1.11e-6) <= 0.11e-6),
        3.4757**2,
        1.446**2,
    )


def solve_i(args):
    """Wrapper of modesovlers"""
    wvl, x, y, epsfunc, boundary, neigs, tol = args
    res_obj = EMpy.modesolvers.FD.VFDModeSolver(wvl, x, y, epsfunc, boundary).solve(
        neigs, tol
    )
    return res_obj


def simulate(wvl_array, x, y, epsfunc, boundary, neigs, tol, num_process):
    """simulations using Pool"""

    pool = Pool(num_process)
    tasks = [(wvl, x, y, epsfunc, boundary, neigs, tol) for wvl in wvl_array]
    solvings = pool.map(solve_i, tasks)

    return solvings


if __name__ == "__main__":
    wl = 1.55e-6
    x = np.linspace(0, 2.48e-6, 125)
    y = np.linspace(0, 2.22e-6, 112)

    neigs = 2
    tol = 1e-8
    boundary = "0000"

    wvl_l = 1.54e-6
    wvl_u = 1.56e-6
    num_wvl = 20

    wvls = np.linspace(wvl_l, wvl_u, num_wvl)
    # print(wvls)
    runtimes = []
    pocket_res = []

    for wvl in wvls:
        start = time.time()
        args = (wvl, x, y, epsfunc, boundary, neigs, tol)
        res = solve_i(args)
        # solver = EMpy.modesolvers.FD.VFDModeSolver(wvl, x, y, epsfunc, boundary).solve(
        #     neigs, tol
        # )
        stop = time.time()
        runtimes.append(stop - start)
        pocket_res.append(res)

    total_time = np.round(np.array(runtimes).sum(), 2)
    print("Time for solving {} wvls in series is {} sec".format(len(wvls), total_time))

    ## multiprocess runtime testing
    processer = 4
    start = time.time()
    simulations = simulate(wvls, x, y, epsfunc, boundary, neigs, tol, processer)
    stop = time.time()
    total_time_MP = np.round((stop - start), 2)
    print("MP takes {} sec".format(total_time_MP))
