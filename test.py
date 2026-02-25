from team_4 import MyAgent
from competition import Competition


import numpy as np
import random


# -----------------------------
# Utility
# -----------------------------

def rand(a=0, b=10):
    return random.uniform(a, b)


# -----------------------------
# 1️⃣ Zero-Sum (Matching Pennies type)
# A + B = 0, no pure Nash, mixed only
# -----------------------------

def gen_zero_sum():
    x = rand(1, 5)
    y = rand(1, 5)

    A = np.array([[ x, -x],
                  [-y,  y]])

    B = -A
    return A, B


# -----------------------------
# 2️⃣ Mixed-only (non-zero-sum)
# No pure NE, one mixed NE
# -----------------------------
def gen_mixed_only():
    """
    Generates a 2x2 mixed-only game (no pure NE, one mixed NE)
    using best-response construction.
    """
    high = rand(5, 10)
    low = rand(0, 3)

    A = np.array([[high, low],[low, high]])
    B = np.array([[low, high],[high, low]])
    return A, B

# -----------------------------
# 3️⃣ Coordination
# Two diagonal pure Nash equilibria
# -----------------------------

def gen_coordination():
    high1 = rand(5, 10)
    high2 = rand(5, 10)
    low = rand(0, 3)

    A = np.array([[high1, low],
                  [low,  high2]])

    B = np.array([[high1, low],
                  [low,  high2]])

    return A, B


# -----------------------------
# 4️⃣ Chicken (Anti-coordination)
# Two off-diagonal pure Nash equilibria
# -----------------------------

def gen_anti_coordination():
    win = rand(5, 10)
    lose = rand(1, 3)
    crash = rand(-5, -1)

    A = np.array([[crash, win],
                  [lose,  0]])

    B = np.array([[crash, lose],
                  [win,   0]])

    return A, B


# -----------------------------
# 5️⃣ Prisoner's Dilemma
# Dominant strategy for both, inefficient equilibrium
# T > R > P > S
# -----------------------------

def gen_prisoners_dilemma():
    T = rand(6, 10)
    R = rand(4, 5)
    P = rand(2, 3)
    S = rand(0, 1)

    A = np.array([[R, S],
                  [T, P]])

    B = np.array([[R, T],
                  [S, P]])

    return A, B


# -----------------------------
# 6️⃣ Deadlock
# Dominant strategy for both, efficient equilibrium
# -----------------------------

def gen_deadlock():
    best = rand(6, 10)
    mid = rand(3, 5)
    low = rand(0, 2)

    A = np.array([[mid, low],
                  [best, best]])

    B = np.array([[mid, best],
                  [low, best]])

    return A, B


# -----------------------------
# 7️⃣ Harmony
# Single pure Nash, Pareto efficient, no dilemma
# -----------------------------

def gen_harmony():
    best = rand(6, 10)
    alt = rand(3, 5)
    low = rand(0, 2)
    eps = 10**(-3)  # small epsilon to ensure strict inequalities
    A = np.array([[best, alt-eps],
                  [low,  alt]])

    B = np.array([[best, low],
                  [alt-eps,  alt]])

    return A, B


# -----------------------------
# 8️⃣ General 2x2 (fallback)
# Completely random
# -----------------------------

def gen_general():
    A = np.random.uniform(0, 10, (2, 2))
    B = np.random.uniform(0, 10, (2, 2))
    return A, B


def gen_symmetric_shuffle():
    a, b, c, d = rand(0, 10), rand(0, 10), rand(0, 10), rand(0, 10)
    # get two random orders of the payoffs
    order1 = np.random.permutation([a, b, c, d])
    order2 = np.random.permutation([a, b, c, d])
    A = np.array([[order1[0], order1[1]],
                  [order1[2], order1[3]]])
    B = np.array([[order2[0], order2[1]],
                  [order2[2], order2[3]]])
    return A, B

# -----------------------------
# Unified interface
# -----------------------------

generators = {
        "zero_sum": gen_zero_sum,
        "mixed": gen_mixed_only,
        "coordination": gen_coordination,
        "anti_coordination": gen_anti_coordination,
        "prisoners_dilemma": gen_prisoners_dilemma,
        "deadlock": gen_deadlock,
        "harmony": gen_harmony,
        "general": gen_general,
        "symetric_shuffle": gen_symmetric_shuffle,
    }

def generate_game(game_type):
    if game_type not in generators:
        raise ValueError(f"Unknown game type: {game_type}")
    return generators[game_type]()



def combine_AB(A, B):
    #combining the matrices such that the 3rd dimension correpsonds to the player/matrix
    return np.stack((A, B), axis=-1)


def summary_test(result):
    from collections import Counter
    counts = Counter(result)
    print("Classification Summary:")
    for classification, count in counts.items():
        print(f"{classification}: {count}")

def test_analysis():
    N=1
    for gametype in generators.keys():
        results = []
        print(f"--- Testing game type: {gametype} ---")
        for i in range(N):
            A, B = generate_game(gametype)
            game = combine_AB(A, B) 
            if N==1: print(f"Game:\nA:\n{A}\nB:\n{B}")
            agent = MyAgent(0, game)
            results.append(agent._classify_game(verbose= True if N==1 else False))
        summary_test(results)
        print(1*"\n")      



def test_strategy(gametype, N, baseline = False, only_row =True):
    from competition import Competition
    from example_agents import AlwaysZero, AlwaysOne, RandomPlayer, TitForTat, WinStayLoseShift, BestResponder
    from team_4 import MixedNEAgent
    agents = { "AlwaysZero": AlwaysZero,
               "AlwaysOne": AlwaysOne,
                "RandomPlayer": RandomPlayer,
                "TitForTat": TitForTat, 
                "WinStayLoseShift": WinStayLoseShift, 
                "BestResponder": BestResponder,
                "MixedNEAgent": MixedNEAgent,
                }
    
    results = {}
    results_baseline = {}
    for name, opponent_cls in agents.items():
        results_local =[]
        results_local_baseline = []
        for _ in range(N):
            if N==1: print(f"Testing against {name}...")
            game = combine_AB(*generate_game(gametype))
            competition = Competition(game)
            if only_row: 
                result = competition.run(agent_class_0 = MyAgent, 
                                         agent_class_1= opponent_cls)
                results_local.append(result["scores"][0])  # Score of our agent (row player)
            else: 
                raise NotImplementedError("Testing both row and column agents not implemented yet, set only_row=True to test row agents only")
            if baseline:
                competition_baseline = Competition(game)
                result_baseline = competition_baseline.run(agents[baseline], opponent_cls)
                results_local_baseline.append(result_baseline["scores"][0])  # Score of TitForTat (row player)
        results[name] = results_local
        results_baseline[name] = results_local_baseline
            

    print(f"Results of MyAgent against various opponents in {gametype} game:")
    for opponent, scores in results.items():
        avg_score = np.mean(scores)
        addition = " --{}: {:.2f} ({:.2f})".format(baseline, np.mean(results_baseline[opponent]), np.std(results_baseline[opponent])) if baseline else ""
        print(f"Against {opponent:<20} | Avg Score = {avg_score:>6.2f} ({np.std(scores):>6.2f}) "+ addition)
    return results, results_baseline

"""
def test_against_self(N=20, rounds=100):
    scores = []
    for _ in range(N):
        A, B = generate_game("anti_coordination")
        game = combine_AB(A, B)
        comp = Competition(game, num_rounds=rounds)
        res = comp.run(MyAgent, MyAgent, verbose=False)
        scores.append(res["scores"])
    print("Avg scores (row,col):",
          (sum(s[0] for s in scores)/N, sum(s[1] for s in scores)/N))

if __name__ == "__main__":
    test_against_self(N=50, rounds=100)
"""


if __name__ == "__main__":
    test_strategy("deadlock", N=100, baseline = "BestResponder")

    


