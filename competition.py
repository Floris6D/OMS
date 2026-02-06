"""
Bimatrix Game Competition Framework
====================================
VU Amsterdam - Game Theory Course
Master Econometrics and Operations Research

This framework runs repeated bimatrix game competitions between two agents.
Students implement the Agent class to participate in the tournament.
"""

import numpy as np
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import importlib.util
import sys


class Agent(ABC):
    """
    Base class for tournament agents. Students must extend this class.
    
    The agent plays a repeated 2x2 bimatrix game. In each round:
    1. get_action() is called to request your action (0 or 1)
    2. observe_result() is called to inform you of the outcome
    
    You have access to the full payoff matrix and know whether you are
    the row player (player_id=0) or column player (player_id=1).
    """
    
    def __init__(self, player_id: int, payoff_matrix: np.ndarray):
        """
        Initialize the agent.
        
        Parameters
        ----------
        player_id : int
            0 if you are the row player, 1 if you are the column player
        payoff_matrix : np.ndarray
            A 2x2x2 array where payoff_matrix[i, j, k] is the payoff
            to player k when row player chooses action i and column player 
            chooses action j.
            
            Example for Prisoner's Dilemma:
                payoff_matrix[0, 0, :] = [3, 3]  # Both cooperate
                payoff_matrix[0, 1, :] = [0, 5]  # Row cooperates, Column defects
                payoff_matrix[1, 0, :] = [5, 0]  # Row defects, Column cooperates
                payoff_matrix[1, 1, :] = [1, 1]  # Both defect
                
        """
        self.player_id = player_id
        self.payoff_matrix = payoff_matrix
        
        # Extract your own payoff matrix for convenience
        # my_payoffs[i, j] = my payoff when row plays i, column plays j
        self.my_payoffs = payoff_matrix[:, :, player_id]
        self.opp_payoffs = payoff_matrix[:, :, 1 - player_id]
        
        # History tracking (students may use or ignore)
        self.history: List[Tuple[int, int, float, float]] = []
    
    @abstractmethod
    def get_action(self) -> int:
        """
        Choose an action for this round.
        
        Returns
        -------
        int
            Your action: 0 or 1
            - If you are row player (player_id=0): this selects the row
            - If you are column player (player_id=1): this selects the column
        """
        pass
    
    def observe_result(self, my_action: int, opp_action: int, 
                       my_payoff: float, opp_payoff: float) -> None:
        """
        Called after each round to inform you of the outcome.
        
        Override this method if you want to implement learning/adaptation.
        The default implementation stores results in self.history.
        
        Parameters
        ----------
        my_action : int
            The action you played (0 or 1)
        opp_action : int
            The action your opponent played (0 or 1)
        my_payoff : float
            Your payoff this round
        opp_payoff : float
            Your opponent's payoff this round
        """
        self.history.append((my_action, opp_action, my_payoff, opp_payoff))
    
    def get_my_payoff(self, my_action: int, opp_action: int) -> float:
        """
        Utility method: get your payoff for a given action combination.
        
        Parameters
        ----------
        my_action : int
            Your action (0 or 1)
        opp_action : int
            Opponent's action (0 or 1)
            
        Returns
        -------
        float
            Your payoff
        """
        if self.player_id == 0:  # Row player
            return self.payoff_matrix[my_action, opp_action, 0]
        else:  # Column player
            return self.payoff_matrix[opp_action, my_action, 1]


class Competition:
    """
    Runs a repeated bimatrix game competition between two agents.
    """
    
    def __init__(self, payoff_matrix: np.ndarray, num_rounds: int = 100,
                 prob_continue: Optional[float] = None, seed: Optional[int] = None):
        """
        Initialize a competition.
        
        Parameters
        ----------
        payoff_matrix : np.ndarray
            2x2x2 payoff marix
        num_rounds : int
            Number of rounds to play
        """
        self.payoff_matrix = payoff_matrix
        self.num_rounds = num_rounds
        
    def run(self, agent_class_0: type, agent_class_1: type, 
            verbose: bool = False) -> dict:
        """
        Run a competition between two agent classes.
        
        Parameters
        ----------
        agent_class_0 : type
            Class for the row player (must extend Agent)
        agent_class_1 : type
            Class for the column player (must extend Agent)
        verbose : bool
            If True, print round-by-round results
            
        Returns
        -------
        dict
            Results containing:
            - 'scores': tuple of (agent_0_score, agent_1_score)
            - 'history': list of (action_0, action_1, payoff_0, payoff_1) tuples
            - 'rounds_played': number of rounds played
        """
        
        # Initialize agents
        agent_0 = agent_class_0(
            player_id=0, 
            payoff_matrix=self.payoff_matrix.copy()
        )
        agent_1 = agent_class_1(
            player_id=1, 
            payoff_matrix=self.payoff_matrix.copy()
        )
        
        # Track results
        history = []
        score_0 = 0.0
        score_1 = 0.0
        
        if verbose:
            print(f"{'Round':<6} {'A0':<4} {'A1':<4} {'P0':<8} {'P1':<8} {'Total0':<10} {'Total1':<10}")
            print("-" * 60)
        
        # Play rounds
        for round_num in range(self.num_rounds):
            # Get actions
            action_0 = agent_0.get_action()
            action_1 = agent_1.get_action()
            
            # Validate actions
            if action_0 not in [0, 1]:
                raise ValueError(f"Agent 0 returned invalid action: {action_0}")
            if action_1 not in [0, 1]:
                raise ValueError(f"Agent 1 returned invalid action: {action_1}")
            
            # Compute payoffs
            payoff_0 = self.payoff_matrix[action_0, action_1, 0]
            payoff_1 = self.payoff_matrix[action_0, action_1, 1]
            
            # Update scores
            score_0 += payoff_0
            score_1 += payoff_1
            
            # Record history
            history.append((action_0, action_1, payoff_0, payoff_1))
            
            if verbose:
                print(f"{round_num+1:<6} {action_0:<4} {action_1:<4} {payoff_0:<8.2f} {payoff_1:<8.2f} {score_0:<10.2f} {score_1:<10.2f}")
            
            # Inform agents of results
            agent_0.observe_result(action_0, action_1, payoff_0, payoff_1)
            agent_1.observe_result(action_1, action_0, payoff_1, payoff_0)
        
        if verbose:
            print("-" * 60)
            print(f"Final scores: Agent 0 = {score_0:.2f}, Agent 1 = {score_1:.2f}")
        
        return {
            'scores': (score_0, score_1),
            'history': history,
            'rounds_played': self.num_rounds
        }


def load_game(filepath: str) -> np.ndarray:
    """
    Load a bimatrix game from a text file.
    
    File format:
    - Lines starting with # are comments
    - Each non-comment line contains one row of the bimatrix
    - Each cell is formatted as "row_payoff,col_payoff"
    - Cells are separated by whitespace
    
    Example file:
    ```
    # Chicken
    # Action 0 = Stop, Action 1 = Drive
    1,1 0,2
    2,0 -10,-10
    ```
    
    Parameters
    ----------
    filepath : str
        Path to the game file
        
    Returns
    -------
    np.ndarray
        Shape (2, 2, 2) payoff matrix
    """
    rows = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse cells
            cells = line.split()
            row_payoffs = []
            col_payoffs = []
            
            for cell in cells:
                parts = cell.split(',')
                if len(parts) != 2:
                    raise ValueError(f"Invalid cell format: {cell}. Expected 'row_payoff,col_payoff'")
                row_payoffs.append(float(parts[0]))
                col_payoffs.append(float(parts[1]))
            
            rows.append((row_payoffs, col_payoffs))
    
    if len(rows) != 2:
        raise ValueError(f"Expected 2 rows, got {len(rows)}")
    if len(rows[0][0]) != 2 or len(rows[1][0]) != 2:
        raise ValueError("Expected 2 columns per row")
    
    # Build payoff matrix
    payoff_matrix = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            payoff_matrix[i, j, 0] = rows[i][0][j]  # Row player payoff
            payoff_matrix[i, j, 1] = rows[i][1][j]  # Column player payoff
    
    return payoff_matrix


def load_agent_class(filepath: str, class_name: str = "MyAgent") -> type:
    """
    Dynamically load an agent class from a Python file.
    
    Parameters
    ----------
    filepath : str
        Path to the Python file containing the agent class
    class_name : str
        Name of the agent class to load (default: "MyAgent")
        
    Returns
    -------
    type
        The agent class
    """
    spec = importlib.util.spec_from_file_location("agent_module", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, class_name):
        raise ValueError(f"Agent file does not contain class '{class_name}'")
    
    return getattr(module, class_name)

if __name__ == "__main__":
    # Demo: run a simple competition
    print("=" * 60)
    print("Bimatrix Game Competition Framework - Demo")
    print("=" * 60)
    
    # Load game from a file
    game = load_game("game.txt")
    print("\nPayoff Matrix:")
    print("Action 0, Action 1")
    print(f"  (C,C): Row={game[0,0,0]}, Col={game[0,0,1]}")
    print(f"  (C,D): Row={game[0,1,0]}, Col={game[0,1,1]}")
    print(f"  (D,C): Row={game[1,0,0]}, Col={game[1,0,1]}")
    print(f"  (D,D): Row={game[1,1,0]}, Col={game[1,1,1]}")
	
	# Load agents
    try:
        agent_class_1 = load_agent_class("example_agents.py", "AlwaysZero")
    except Exception as e:
        print(f"Error loading agent 1: {e}", file=sys.stderr)
        raise e
    
    try:
        agent_class_2 = load_agent_class("example_agents.py", "AlwaysOne")
    except Exception as e:
        print(f"Error loading agent 2: {e}", file=sys.stderr)
        raise e
	
	
    # Run competition
    print("\n" + "=" * 60)
    print("Competition: AlwaysZero (Row) vs AlwaysOne (Column)")
    print("=" * 60)
    
    comp = Competition(game, 10)
    results = comp.run(agent_class_1, agent_class_2, verbose=True)
    
    print(f"\nAlwaysZero scored: {results['scores'][0]:.2f}")
    print(f"AlwaysOne scored: {results['scores'][1]:.2f}")
