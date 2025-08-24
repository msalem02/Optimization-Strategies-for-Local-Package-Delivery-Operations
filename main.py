# Mohammad Salem - 1203022 - Sec. 3

import sys
import csv
import math
import random
import copy
from dataclasses import dataclass
from typing import List, Tuple
import tkinter as tk
from tkinter import scrolledtext, messagebox as mb

# ----------------------------
# CONFIGURATION
# ----------------------------
CSV_FILE = "input.csv"            # Path to CSV

# Simulated Annealing defaults
SA_COOLING_RATE   = 0.95
SA_INITIAL_TEMP   = 1000.0
SA_STOP_TEMP      = 1.0
SA_ITERS_PER_TEMP = 100

# Genetic Algorithm defaults
GA_POP_SIZE       = 80
GA_MUTATION_RATE  = 0.05
GA_GENERATIONS    = 500
GA_ELITISM        = 2

@dataclass
class Package:
    id: int
    x: float
    y: float
    weight: float
    priority: int

@dataclass
class Vehicle:
    id: int
    capacity: float

@dataclass
class Problem:
    depot: Tuple[float, float]
    packages: List[Package]
    vehicles: List[Vehicle]


def load_packages_from_csv(path: str) -> List[Package]:
    packages: List[Package] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k.strip().lower(): v for k, v in raw.items()}
            packages.append(Package(
                id=int(row['id']),
                x=float(row['x']),
                y=float(row['y']),
                weight=float(row['weight']),
                priority=int(row['priority'])
            ))
    return packages


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_distance(route: List[int], problem: Problem) -> float:
    dist = 0.0
    prev = problem.depot
    id2pkg = {p.id: p for p in problem.packages}
    for pid in route:
        pkg = id2pkg[pid]
        curr = (pkg.x, pkg.y)
        dist += euclidean(prev, curr)
        prev = curr
    dist += euclidean(prev, problem.depot)
    return dist


def total_distance(all_routes: List[List[int]], problem: Problem) -> float:
    return sum(route_distance(r, problem) for r in all_routes)


def two_opt(route: List[int], problem: Problem) -> List[int]:
    best = route[:]
    best_cost = route_distance(best, problem)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = route_distance(new_route, problem)
                if new_cost + 1e-6 < best_cost:
                    best, best_cost = new_route, new_cost
                    improved = True
                    break
            if improved:
                break
    return best


# ----------------------------
# Simulated Annealing
# ----------------------------
class SAParams:
    def __init__(self,
                 cooling_rate: float,
                 T0: float = SA_INITIAL_TEMP,
                 stop_temp: float = SA_STOP_TEMP,
                 iters_per_temp: int = SA_ITERS_PER_TEMP):
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.stop_temp = stop_temp
        self.iters_per_temp = iters_per_temp


def initial_solution(problem: Problem) -> List[List[int]]:
    sorted_pkgs = sorted(
        problem.packages,
        key=lambda p: (p.priority, euclidean((p.x, p.y), problem.depot))
    )
    routes = [[] for _ in problem.vehicles]
    caps = [v.capacity for v in problem.vehicles]
    for pkg in sorted_pkgs:
        for i in range(len(routes)):
            if caps[i] >= pkg.weight:
                routes[i].append(pkg.id)
                caps[i] -= pkg.weight
                break
    return routes


def get_neighbor(routes: List[List[int]], problem: Problem) -> List[List[int]]:
    new = copy.deepcopy(routes)
    if random.random() < 0.5 and len(routes) >= 2:
        # swap between vehicles
        v1, v2 = random.sample(range(len(routes)), 2)
        if new[v1] and new[v2]:
            i = random.randrange(len(new[v1]))
            j = random.randrange(len(new[v2]))
            new[v1][i], new[v2][j] = new[v2][j], new[v1][i]
    else:
        # swap within a route
        v = random.randrange(len(routes))
        if len(new[v]) >= 2:
            i, j = random.sample(range(len(new[v])), 2)
            new[v][i], new[v][j] = new[v][j], new[v][i]
    return new


def simulated_annealing(problem: Problem, params: SAParams):
    current = initial_solution(problem)
    current = [two_opt(r, problem) for r in current]
    best = current[:]
    best_cost = total_distance(best, problem)
    cost = best_cost
    T = params.T0

    while T > params.stop_temp:
        for _ in range(params.iters_per_temp):
            neigh = get_neighbor(current, problem)
            neigh = [two_opt(r, problem) for r in neigh]
            c = total_distance(neigh, problem)
            if c < cost or random.random() < math.exp((cost - c) / T):
                current, cost = neigh, c
                if cost < best_cost:
                    best, best_cost = current[:], cost
        T *= params.cooling_rate

    return best, best_cost


# ----------------------------
# Genetic Algorithm
# ----------------------------
class GAParams:
    def __init__(self,
                 pop_size: int,
                 mutation_rate: float,
                 generations: int = GA_GENERATIONS):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations


def _random_solution(problem: Problem) -> List[List[int]]:

    pkgs = problem.packages[:]
    random.shuffle(pkgs)
    routes = [[] for _ in problem.vehicles]
    caps = [v.capacity for v in problem.vehicles]

    # Phase 1
    for pkg in pkgs:
        choices = [i for i in range(len(routes)) if caps[i] >= pkg.weight]
        if choices:
            i = random.choice(choices)
            routes[i].append(pkg.id)
            caps[i] -= pkg.weight

    # Phase 2
    assigned = {pid for r in routes for pid in r}
    for pkg in problem.packages:
        if pkg.id not in assigned:
            best_i = max(range(len(routes)), key=lambda i: caps[i])
            routes[best_i].append(pkg.id)
            caps[best_i] -= pkg.weight
            assigned.add(pkg.id)

    return routes


def _repair_solution(routes: List[List[int]], problem: Problem) -> List[List[int]]:

    id2pkg = {p.id: p for p in problem.packages}
    seen = set()
    new = [[] for _ in routes]
    caps = [v.capacity for v in problem.vehicles]


    for vi, r in enumerate(routes):
        for pid in r:
            if pid not in seen and caps[vi] >= id2pkg[pid].weight:
                new[vi].append(pid)
                seen.add(pid)
                caps[vi] -= id2pkg[pid].weight


    for pkg in problem.packages:
        if pkg.id not in seen:
            best_i = max(range(len(new)), key=lambda i: caps[i])
            new[best_i].append(pkg.id)
            caps[best_i] -= pkg.weight
            seen.add(pkg.id)

    return new


def genetic_algorithm(problem: Problem, params: GAParams):
    # seed + random initial population
    seed = initial_solution(problem)
    pop = [seed] + [_random_solution(problem) for _ in range(params.pop_size - 1)]
    # local refine all
    pop = [[two_opt(r, problem) for r in ind] for ind in pop]
    costs = [total_distance(ind, problem) for ind in pop]

    for _ in range(params.generations):
        # rank
        ranked = sorted(zip(costs, pop), key=lambda x: x[0])
        pop = [ind for _, ind in ranked]
        costs = [c for c, _ in ranked]

        new_pop = pop[:GA_ELITISM]
        while len(new_pop) < params.pop_size:
            # tournament selection
            i, j = random.sample(range(len(pop)), 2)
            p1 = pop[i] if costs[i] < costs[j] else pop[j]
            k, l = random.sample(range(len(pop)), 2)
            p2 = pop[k] if costs[k] < costs[l] else pop[l]

            # crossover
            child = []
            for vi in range(len(problem.vehicles)):
                if random.random() < 0.5:
                    child.append(p1[vi][:])
                else:
                    child.append(p2[vi][:])

            # repair & refine
            child = _repair_solution(child, problem)
            child = [two_opt(r, problem) for r in child]

            # mutation
            if random.random() < params.mutation_rate:
                idx = random.randrange(len(child))
                child[idx] = two_opt(child[idx], problem)

            new_pop.append(child)

        pop = new_pop
        costs = [total_distance(ind, problem) for ind in pop]

    # final best
    best = pop[0]
    best = [two_opt(r, problem) for r in best]
    best_cost = total_distance(best, problem)
    return best, best_cost


# ----------------------------
# GUI Display
# ----------------------------
def show_solution_gui(alg_name, total_cost, solution, problem, capacities, overloaded):
    id2pkg = {p.id: p for p in problem.packages}
    root = tk.Tk()
    root.title("Routing Solution Viewer")
    text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
    text.pack(padx=10, pady=10)

    text.insert(tk.END, f"Algorithm: {alg_name}\n")
    text.insert(tk.END, f"Total distance (all vehicles): {total_cost:.2f} km\n\n")

    if overloaded:
        text.insert(tk.END, f"Overloaded (not assigned): {overloaded}\n\n")

    for vidx, route in enumerate(solution):
        path = [problem.depot] + [(id2pkg[pid].x, id2pkg[pid].y) for pid in route] + [problem.depot]
        dist = route_distance(route, problem)
        path_str = " -> ".join(f"({x:.1f},{y:.1f})" for x, y in path)

        text.insert(tk.END, f"Vehicle #{vidx+1} (cap={capacities[vidx]} kg):\n")
        text.insert(tk.END, f"  Package IDs: {route}\n")
        text.insert(tk.END, f"  Path: {path_str}\n")
        text.insert(tk.END, f"  Route distance: {dist:.2f} km\n\n")

    tk.Button(root, text="Close", command=root.destroy).pack(pady=5)
    root.mainloop()


# ----------------------------
# Main
# ----------------------------
def run():
    try:
        packages = load_packages_from_csv(CSV_FILE)
    except Exception as e:
        mb.showerror("File Error", str(e))
        sys.exit(1)

    # console input for vehicles
    try:
        n = int(input("Enter number of vehicles: "))
        capacities = [float(input(f"Capacity of vehicle #{i+1} (kg): "))
                      for i in range(n)]
    except:
        mb.showerror("Input Error", "Please enter valid numbers.")
        sys.exit(1)

    max_cap = max(capacities)
    overloaded = [p.id for p in packages if p.weight > max_cap]
    feasible = [p for p in packages if p.weight <= max_cap]

    vehicles = [Vehicle(i, capacities[i]) for i in range(n)]
    problem = Problem((0.0, 0.0), feasible, vehicles)

    def on_select():
        sel = var.get()
        if sel == "Simulated Annealing":
            sol, cost = simulated_annealing(problem, SAParams(SA_COOLING_RATE))
        elif sel == "Genetic Algorithm":
            sol, cost = genetic_algorithm(problem, GAParams(GA_POP_SIZE, GA_MUTATION_RATE))
        else:
            root.destroy()
            sys.exit(0)
        root.destroy()
        show_solution_gui(sel, cost, sol, problem, capacities, overloaded)

    root = tk.Tk()
    root.title("Select Optimization Algorithm")
    tk.Label(root, text="Choose Optimization Algorithm:", font=("Arial", 12)).pack(pady=10)
    var = tk.StringVar(value="Simulated Annealing")
    tk.OptionMenu(root, var, "Simulated Annealing", "Genetic Algorithm", "Exit").pack(pady=5)
    tk.Button(root, text="Run", command=on_select).pack(pady=10)
    root.mainloop()


if __name__ == "__main__":
    run()