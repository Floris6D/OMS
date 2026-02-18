"""
Template Agent for Bimatrix Game Tournament
============================================

Instructions:
1. Rename this file to team_[YourTeamNumber].py
2. Implement the analyze_game() method to help you classify the game
3. Implement the get_action() method and any helper methods you want to use
4. Optionally override observe_result() for learning/adapting strategies
5. Test against the example agents before submission. Implement the MixedNEAgent agent as a potential agent to test again
"""

import numpy as np
from competition import Agent

def create_empty_analysis() -> dict:
	"""
	Creates an empty game analysis dictionary with all required fields.
	
	Your `analyze_game()` function must return a dictionary with exactly
	these keys, filled with the correct values for the given game.
	
	Returns
	-------
	dict
		Empty analysis dictionary with None/default values
	"""
	return {
		# ============================================================
		# DOMINANCE ANALYSIS
		# ============================================================
		# Strictly weakly strategy
		# Value: 0, 1, or None (if no weakly dominant strategy exists)
		
		'row_weakly_dominant': None,  # Row player's weakly dominant action
		'col_weakly_dominant': None,  # Column player's weakly dominant action
		
		# ============================================================
		# NASH EQUILIBRIUM ANALYSIS
		# ============================================================
		# Pure strategy Nash equilibria
		# Value: List of tuples [(i, j), ...] where i=row action, j=col action
		# Example: [(0, 0), (1, 1)] means two pure Nash equilibria
		
		'pure_nash': [],
		
		# Mixed strategy Nash equilibrium probabilities.
		# Value: Tuple (p, q) where:
		#	p = probability that ROW player plays action 0
		#	q = probability that COLUMN player plays action 0
		# Set to None if only pure equilibria exist 
		# (i.e., if in the equilibrium both players play their strategy with p=1 or p=0 and q=1 or q=0)
		
		'mixed_nash': None,
		
		# Total number of Nash equilibria (counting pure and mixed separately)
		'num_nash': 0,
		
		# Is there exactly one Nash equilibrium?
		'unique_nash': False,
		
		# ============================================================
		# PARETO EFFICIENCY ANALYSIS
		# ============================================================
		# Pareto optimal outcomes: no other outcome makes both players
		# at least as well off and at least one player strictly better off.
		# Value: List of tuples [(i, j), ...]
		
		'pareto_optimal_outcomes': [],
		
		# Is at least one Nash equilibrium Pareto optimal?
		'nash_pareto_optimal': False,
		
		# ============================================================
		# WELFARE ANALYSIS
		# ============================================================
		# Social welfare matrix: sum of both players' payoffs at each outcome
		# Value: 2x2 numpy array where entry [i,j] = A[i,j] + B[i,j]
		
		'social_welfare': None,
		
		# Social optimum: outcome(s) maximizing total welfare
		# Value: List of tuples [(i, j), ...]
		
		'social_optimum': [],
		
		# Price of Anarchy
		# Note: This concept will be introduced in week 3. It may not be
		# necessary for the strategy selection, but gives insight into
		# the type of game nonetheless.
		
		'price_of_anarchy': None,
			
		# ============================================================
		# STRATEGIC STRUCTURE
		# ============================================================
		# Coordination game: both players prefer if the opponent plays
		# the same action as themself over the opponent playing a
		# different action (Example: Battle of Sexes)
		# Value: bool
		
		'is_coordination': False,
		
		# Anti-coordination game: both players prefer if the opponent plays a different
		# action then themself over the opponent playing the same action
		# Example: Chicken Game (note: Not the case in the Grade Game, as if a player
		# plays 'Confess' they prefer the opponent to 'Deny', while if a player
		# plays 'Deny' they rather want the opponent to 'Deny' as well.
		# Value: bool
		
		'is_anti_coordination': False
	}


def analyze_game(A: np.ndarray, B: np.ndarray) -> dict:
	"""
	Analyze a 2x2 bimatrix game and fill the analysis dictionary.
	
	============================================================
	YOU MUST IMPLEMENT THIS FUNCTION
	============================================================
	
	Parameters
	----------
	A : np.ndarray, row player's 2x2 payoff matrix.
		A[i, j] = row player's payoff when row plays i, column plays j
		
	B : np.ndarray, column player's 2x2 payoff matrix.
		B[i, j] = column player's payoff when row plays i, column plays j
	
	Returns
	-------
	dict
		Completed analysis dictionary with all fields filled.
		See `	()` for the required fields.
	
	Example
	-------
	For Grade Game: Action 0: Confess, Action 1: Deny
		A = [[6, 9], [0, 7]]  # Row player payoffs
		B = [[6, 0], [0, 7]]  # Column player payoffs
	
	Your function should return:
		{
			'row_strictly_dominant': 0,	 # Confess weakly dominates
			'col_strictly_dominant': 0,	 # Confess weakly dominates
			'pure_nash': [(0, 0)],	# (Confess, Confess)
			'mixed_nash': None,	 # No real-valued mixed equilibrium
			'num_nash': 1,
			'unique_nash': True,
			'pareto_optimal_outcomes': [(0, 1), (1, 0), (1,1)],
			'nash_pareto_optimal': False,  # (0,0) = (Confess,Confess) is not Pareto optimal
			... etc
		}
	"""
	analysis = create_empty_analysis()
	
	# ============================================================
	# TODO: IMPLEMENT YOUR ANALYSIS HERE
	# ============================================================
	
	# Hint: Start with dominance analysis
	# Row player: action 0 weakly dominates action 1 if A[0,j] >= A[1,j] for all j
	
	# Hint: For Nash equilibria, check pure strategies first
	# (i,j) is pure Nash if i is best response to j AND j is best response to i
	
	# Hint: For mixed Nash, derive from indifference conditions
	# Row indifferent: q*A[0,0] + (1-q)*A[0,1] = q*A[1,0] + (1-q)*A[1,1]


	A = np.asarray(A, dtype=float)
	B = np.asarray(B, dtype=float)

	if A.shape != (2, 2) or B.shape != (2, 2):
		raise ValueError("analyze_game expects A and B to be 2x2 matrices.")

	# ============================================================
	# DOMINANCE ANALYSIS
	# ============================================================
	row0_dom = (A[0, 0] >= A[1, 0]) and (A[0, 1] >= A[1, 1])
	row1_dom = (A[1, 0] >= A[0, 0]) and (A[1, 1] >= A[0, 1])
	if row0_dom and not row1_dom:
		analysis["row_weakly_dominant"] = 0
		analysis["row_strictly_dominant"] = 0 if (A[0, 0] > A[1, 0]) and (A[0, 1] > A[1, 1]) else None
	elif row1_dom and not row0_dom:
		analysis["row_weakly_dominant"] = 1
		analysis["row_strictly_dominant"] = 1 if (A[1, 0] > A[0, 0]) and (A[1, 1] > A[0, 1]) else None
	else:
		analysis["row_weakly_dominant"] = None  # none or both
		analysis["row_strictly_dominant"] = None  # if both weakly dominant, then no strictly dominant

	col0_dom = (B[0, 0] >= B[0, 1]) and (B[1, 0] >= B[1, 1])
	col1_dom = (B[0, 1] >= B[0, 0]) and (B[1, 1] >= B[1, 0])
	if col0_dom and not col1_dom:
		analysis["col_weakly_dominant"] = 0
		analysis["col_strictly_dominant"] = 0 if (B[0, 0] > B[0, 1]) and (B[1, 0] > B[1, 1]) else None
	elif col1_dom and not col0_dom:
		analysis["col_weakly_dominant"] = 1
		analysis["col_strictly_dominant"] = 1 if (B[0, 1] > B[0, 0]) and (B[1, 1] > B[1, 0]) else None
	else:
		analysis["col_weakly_dominant"] = None
		analysis["col_strictly_dominant"] = None

	# ============================================================
	# NASH EQUILIBRIUM ANALYSIS
	# ============================================================
	#pure nash equilibria
	pure_nash = []

	# row best responses to each column action j
	row_br = []
	for j in (0, 1):
		col_payoffs = np.array([A[0, j], A[1, j]])
		m = np.max(col_payoffs)
		row_br.append(set(np.where(col_payoffs == m)[0].tolist()))

	# column best responses to each row action i
	col_br = []
	for i in (0, 1):
		row_payoffs = np.array([B[i, 0], B[i, 1]])
		m = np.max(row_payoffs)
		col_br.append(set(np.where(row_payoffs == m)[0].tolist()))

	for i in (0, 1):
		for j in (0, 1):
			if (i in row_br[j]) and (j in col_br[i]):
				pure_nash.append((i, j))

	analysis["pure_nash"] = pure_nash

	# mixed nash
	def interior_prob(val: float) -> bool:
		return (val > 0.0) and (val < 1.0) and np.isfinite(val)

	# q = prob column plays 0 that makes row indifferent between 0 and 1
	den_q = (A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1])
	num_q = (A[1, 1] - A[0, 1])

	# p = prob row plays 0 that makes column indifferent between 0 and 1
	den_p = (B[0, 0] - B[0, 1] - B[1, 0] + B[1, 1])
	num_p = (B[1, 1] - B[1, 0])

	mixed = None
	if den_q != 0 and den_p != 0:
		q = num_q / den_q
		p = num_p / den_p
		if interior_prob(p) and interior_prob(q):
			mixed = (float(p), float(q))

	# set mixed_nash None if equilibrium is pure/boundary
	analysis["mixed_nash"] = mixed

	# total number of nash equilibria + unique?
	analysis["num_nash"] = len(pure_nash) + (1 if mixed is not None else 0)
	analysis["unique_nash"] = (analysis["num_nash"] == 1)

	# ============================================================
	# PARETO EFFICIENCY ANALYSIS
	# ============================================================
	outcomes = [(i, j) for i in (0, 1) for j in (0, 1)]
	pareto = []

	for (i, j) in outcomes:
		dominated = False
		for (k, l) in outcomes:
			if (k, l) == (i, j):
				continue
			if (A[k, l] >= A[i, j]) and (B[k, l] >= B[i, j]) and (
				(A[k, l] > A[i, j]) or (B[k, l] > B[i, j])
			):
				dominated = True
				break
		if not dominated:
			pareto.append((i, j))

	analysis["pareto_optimal_outcomes"] = pareto
	analysis["nash_pareto_optimal"] = any(ne in pareto for ne in pure_nash)

	# ============================================================
	# WELFARE ANALYSIS
	# ============================================================
	# social wefare + social optimum
	W = A + B
	analysis["social_welfare"] = W

	maxW = np.max(W)
	analysis["social_optimum"] = [(i, j) for (i, j) in outcomes if W[i, j] == maxW]

	# price of anarchy
	ne_outcomes = pure_nash[:]  # mixed welfare is not a single outcome, so ignore for PoA
	if len(ne_outcomes) > 0:
		welfare_ne = [W[i, j] for (i, j) in ne_outcomes]
		min_ne = float(np.min(welfare_ne))
		max_social = float(maxW)

		# avoid other ratios when welfare can be <= 0
		if max_social > 0 and min_ne > 0:
			analysis["price_of_anarchy"] = max_social / min_ne
		elif max_social > 0 and min_ne == 0:
			analysis["price_of_anarchy"] = float("inf")
		else:
			analysis["price_of_anarchy"] = None
	else:
		analysis["price_of_anarchy"] = None

	# ============================================================
	# STRATEGIC STRUCTURE
	# ============================================================
	# coordination: matching opponent’s action
	row_coord = (A[0, 0] >= A[0, 1]) and (A[1, 1] >= A[1, 0])
	col_coord = (B[0, 0] >= B[1, 0]) and (B[1, 1] >= B[0, 1])
	analysis["is_coordination"] = bool(row_coord and col_coord)

	# anti-coordination: mismatching opponent’s action
	row_anti = (A[0, 1] >= A[0, 0]) and (A[1, 0] >= A[1, 1])
	col_anti = (B[1, 0] >= B[0, 0]) and (B[0, 1] >= B[1, 1])
	analysis["is_anti_coordination"] = bool(row_anti and col_anti)

	analysis["zero_sum"] = np.allclose(A + B, 0)
	if len(analysis["pure_nash"])==2 and analysis["is_coordination"]:
		nash1, nash2 = analysis["pure_nash"]
		Apref1 = A[nash1] >= A[nash2]
		Bpref1 = B[nash1] >= B[nash2]
		if Apref1 and Bpref1 or (not Apref1 and not Bpref1):
			analysis["harmony"] = True
		else:
			analysis["harmony"] = False
	else:
		analysis["harmony"] = None
	return analysis


class MyAgent(Agent):
	"""
	Your tournament agent.
	
	Available attributes (set by parent class):
	- self.player_id: 0 if row player, 1 if column player
	- self.payoff_matrix: shape (2,2,2), payoff_matrix[i,j,k] = payoff to 
						  player k when row plays i, col plays j
	- self.my_payoffs: shape (2,2), your payoffs [row_action, col_action]
	- self.opp_payoffs: shape (2,2), opponent's payoffs
	- self.game_length: expected number of rounds
	- self.history: list of (my_action, opp_action, my_payoff, opp_payoff)
	
	Useful method:
	- self.get_my_payoff(my_action, opp_action): returns your payoff
	"""
	
	def __init__(self, player_id: int, payoff_matrix: np.ndarray):
		super().__init__(player_id, payoff_matrix)
		
		# ===== YOUR INITIALIZATION CODE HERE =====
		# Analyze the game matrix, classify the game type, etc.
		# 
		# As a starting pont, you should run the analyze_game() function
		# from game_analysis.py to extract some information of the game.
		# Afterwards, you could, e.g., use the classify_game() function within
		# this agent to classify the game and determine your strategy.
		# ==========================================
		
		self.analysis = analyze_game(self.my_payoffs, self.opp_payoffs)
  
		pass
	
	def get_action(self) -> int:
		"""
		Return your action for this round: 0 or 1.
		
		This is called once per round BEFORE you know the opponent's action.
		Use self.history to see previous rounds.
		
		Returns
		-------
		int
			Your action (0 or 1)
		"""
		
		# ===== YOUR STRATEGY CODE HERE =====
		# 
		# Simple example: always play action 0
		# return 0
		#
		# Random example:
		# return np.random.choice([0, 1])
		#
		# Tit-for-Tat example:
		# if not self.history:
		#	  return 0
		# return self.history[-1][1]  # opponent's last action
		#
		# ===================================
		
		# Default: random action (replace this!)
		return np.random.choice([0, 1])
	
	def observe_result(self, my_action: int, opp_action: int,
					   my_payoff: float, opp_payoff: float) -> None:
		"""
		Called after each round with the results.
		
		Override this if you want to implement learning or adaptation.
		The default just stores the result in self.history.
		
		Parameters
		----------
		my_action : int
			The action you played
		opp_action : int
			The action your opponent played
		my_payoff : float
			Your payoff this round
		opp_payoff : float
			Your opponent's payoff this round
		"""
		# Always call parent to maintain history
		super().observe_result(my_action, opp_action, my_payoff, opp_payoff)
			   
		pass
	
	# ===== HELPER METHODS (examples) =====
	# Add your own helper methods below
	
	def _classify_game(self, verbose=False) -> str:
		"""
		Example helper: Classify the game type based on payoff structure.
		
		Returns a string like 'prisoners_dilemma', 'chicken', 'coordination', etc.
		"""
		# TODO: Implement game classification
		if verbose:
			print("Game analysis:")
			for key, value in self.analysis.items():
				print(f"{key}: {value}")
		# Consider: dominant strategies, Nash equilibria, Pareto efficiency
		if len(self.analysis["pure_nash"])>=1: 
			#could be, prisonerd dillema, deadlock
			if len(self.analysis["pure_nash"]) == 1:
				if (
					self.analysis["row_strictly_dominant"] is not None 
					and self.analysis["col_strictly_dominant"] is not None
					and not self.analysis["nash_pareto_optimal"]
				):
					return "prisoners_dilemma"
				elif (
					self.analysis["row_strictly_dominant"] is not None 
					and self.analysis["col_strictly_dominant"] is not None
					and self.analysis["nash_pareto_optimal"]
				):
					return "deadlock" #
				# 1 NE, also dominant strategy for both, 
				# also pareto optimal, so very easy
				else:
					return "unknown_1NE"

				
			elif len(self.analysis["pure_nash"]) == 2:
				# possibilies are: coordination, anti-coordination (chicken), harmony
				if self.analysis["is_coordination"] and self.analysis["harmony"]:
					return "harmony"
				elif self.analysis["is_coordination"]:
					return "coordination"
				elif self.analysis["is_anti_coordination"]:
					return "anti_coordination"
				else:
					return "unknown_2NE"
			
		elif self.analysis["mixed_nash"] is not None:
			if self.analysis["zero_sum"]:
				return f"zero_sum_mixed"
			else:
				return f"mixed"
		else:
			return "unknown"
		
	# ============================================================
	# REQUIRED: get_analysis() method for grading
	# ============================================================
	# The grading system will call this method to extract your analysis.
	# Do not modify this method.
		
	def get_analysis(self) -> dict:
		"""Returns the game analysis dictionary for grading."""
		return self.analysis

class MixedNEAgent(Agent):
	"""
	Agent that always plays the mixed-stragey NE (or the unique pure NE, if it exists)
	"""

	# ===== Implement the agent playing the mixed NE in this class =============
	# 
	# You can make use of the analyze_game function that you have implemented
	# and add any further functions or attricutes in this class 
	# (note: Most likely, you will not need anything in addition)
	#
	# The agent should play its action according to the probabiltiy distribution
	# in the mixed strategy NE (or one action with certain, if there is only one
	# pure-strategy NE)
	#
	# ==========================================================================


	def __init__(self, player_id: int, payoff_matrix: np.ndarray):
		super().__init__(player_id, payoff_matrix)
		
		pass

	def get_action(self) -> int:
		"""
		Return the action of this agent for the curent round: 0 or 1.
			   
		Returns
		-------
		int
			Action of this agent (0 or 1)
		"""
		
		# Default: random action (replace this!)
		return np.random.choice([0, 1])




# ===== TEST YOUR AGENT =====
if __name__ == "__main__":
	from competition import Competition, load_game, load_agent_class
	
	# Test against a simple opponent
	TitForTat = load_agent_class("example_agents.py", "TitForTat")
	
	# Load a game
	game = load_game("games/grade.txt")
	
	# Print your game analysis:
	myAg = MyAgent(0,game)
	print("My agent analyzed the game matrix as follows:")
	print(myAg.get_analysis())
	
	# Run a short test
	print("Testing my agent against TitForTat...")
	comp = Competition(game, num_rounds=20)
	results = comp.run(MyAgent, TitForTat, verbose=True)
	
	print(f"\nMyAgent scored: {results['scores'][0]:.2f}")
	print(f"TitForTat scored: {results['scores'][1]:.2f}")

if __name__ == "__main__":
    from competition import load_game

    for fname in ["battle_of_sexes.txt", "chicken.txt", "grade.txt"]:
        payoff = load_game(f"games/{fname}")     # shape (2,2,2)
        A = payoff[:, :, 0]                      # row payoffs (2,2)
        B = payoff[:, :, 1]                      # col payoffs (2,2)

        print("\nGame:", fname)
        print("A=\n", A)
        print("B=\n", B)
        print(analyze_game(A, B))



