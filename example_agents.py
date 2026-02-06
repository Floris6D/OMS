"""
Example Agents for Testing
==========================

These agents serve as baselines for testing your strategy.
Feel free to add more agents to this file for your testing!
"""

import numpy as np
from competition import Agent


class AlwaysZero(Agent):
    """Always plays action 0. Trivial baseline."""
    def get_action(self) -> int:
        return 0


class AlwaysOne(Agent):
    """Always plays action 1. Trivial baseline."""
    def get_action(self) -> int:
        return 1


class RandomPlayer(Agent):
    """Plays uniformly at random. Simple baseline."""
    def get_action(self) -> int:
        return np.random.choice([0, 1])


class TitForTat(Agent):
    """
    Classic Tit-for-Tat: Start with 1 (which is denying in the grade game, then mirror opponent's last action.)
    Famous strategy from Axelrod's tournament - effective in Grade's game, but needs to be adapted for random
	or different matrices in which the dominated, but Pareto-optimal strategy is in (0,0) instead of (1,1).
    """
    def get_action(self) -> int:
        if not self.history:
            return 1
        return self.history[-1][1]  # Opponent's last action


class WinStayLoseShift(Agent):
    """
    Repeat last action if payoff was good, switch otherwise.
    Also known as Pavlov. Good threshold is median payoff.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = np.median(self.my_payoffs)
        self.last_action = 0
    
    def get_action(self) -> int:
        return self.last_action
    
    def observe_result(self, my_action, opp_action, my_payoff, opp_payoff):
        super().observe_result(my_action, opp_action, my_payoff, opp_payoff)
        if my_payoff >= self.threshold:
            self.last_action = my_action  # Stay
        else:
            self.last_action = 1 - my_action  # Shift


class BestResponder(Agent):
    """
    Plays best response to opponent's empirical action frequencies.
    Uses Laplace smoothing to avoid extreme estimates early on.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opp_counts = [1, 1]  # Laplace smoothing: start with pseudocounts
    
    def get_action(self) -> int:
        total = sum(self.opp_counts)
        opp_prob_0 = self.opp_counts[0] / total
        
        # Expected payoff for each action
        exp_0 = opp_prob_0 * self.my_payoffs[0, 0] + (1 - opp_prob_0) * self.my_payoffs[0, 1]
        exp_1 = opp_prob_0 * self.my_payoffs[1, 0] + (1 - opp_prob_0) * self.my_payoffs[1, 1]
        
        if exp_0 > exp_1:
            return 0
        elif exp_1 > exp_0:
            return 1
        else:
            return np.random.choice([0, 1])
    
    def observe_result(self, my_action, opp_action, my_payoff, opp_payoff):
        super().observe_result(my_action, opp_action, my_payoff, opp_payoff)
        self.opp_counts[opp_action] += 1