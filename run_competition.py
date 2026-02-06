#!/usr/bin/env python3
"""
Command-Line Competition Runner
================================
Run bimatrix game competitions from the command line.

Usage:
    python run_competition.py <game_file> <agent1_file> <agent2_file> [options]
    
Examples:
    python run_competition.py games/grade.txt example_agents.py example_agents.py --class1 TitForTat --class2 BestResponder -v
    python run_competition.py games/chicken.txt template_agent.py example_agents.py --class1 MyAgent --class2 AlwaysOne --rounds 50
"""

import argparse
import sys
from competition import Competition, load_game, load_agent_class


def main():
    parser = argparse.ArgumentParser(
        description="Run a bimatrix game competition between two agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_competition.py game.txt example_agents.py example_agents.py \\
      --class1 TitForTat --class2 BestResponder -v
      
  python run_competition.py game.txt template_agent.py example_agents.py \\
      --class1 MyAgent --class2 AlwaysOne --rounds 50
        """
    )
    
    parser.add_argument("game_file", help="Path to game definition file (.txt)")
    parser.add_argument("agent1_file", help="Path to agent 1 Python file")
    parser.add_argument("agent2_file", help="Path to agent 2 Python file")
    
    parser.add_argument("--class1", default="MyAgent", 
                        help="Class name for agent 1 (default: MyAgent)")
    parser.add_argument("--class2", default="MyAgent",
                        help="Class name for agent 2 (default: MyAgent)")
    
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of rounds (default: 100)")   
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print round-by-round results")
    parser.add_argument("--show-matrix", action="store_true",
                        help="Print the payoff matrix before running")
    
    args = parser.parse_args()
    
    # Load game
    try:
        game = load_game(args.game_file)
    except Exception as e:
        print(f"Error loading game file: {e}", file=sys.stderr)
        return 1
    
    if args.show_matrix:
        print("Payoff Matrix:")
        print("              Col: 0       Col: 1")
        print(f"  Row: 0    ({game[0,0,0]:5.1f},{game[0,0,1]:5.1f})  ({game[0,1,0]:5.1f},{game[0,1,1]:5.1f})")
        print(f"  Row: 1    ({game[1,0,0]:5.1f},{game[1,0,1]:5.1f})  ({game[1,1,0]:5.1f},{game[1,1,1]:5.1f})")
        print()
    
    # Load agents
    try:
        agent_class_1 = load_agent_class(args.agent1_file, args.class1)
    except Exception as e:
        print(f"Error loading agent 1: {e}", file=sys.stderr)
        return 1
    
    try:
        agent_class_2 = load_agent_class(args.agent2_file, args.class2)
    except Exception as e:
        print(f"Error loading agent 2: {e}", file=sys.stderr)
        return 1
    
    # Run competition
    comp = Competition(
        payoff_matrix=game,
        num_rounds=args.rounds,
    )
    
    results = comp.run(agent_class_1, agent_class_2, verbose=args.verbose)
    
    # Print results
    if not args.verbose:
        print(f"\nResults after {results['rounds_played']} rounds:")
    else:
        print()
        
    print(f"  Agent 1 ({args.class1}): {results['scores'][0]:.2f}")
    print(f"  Agent 2 ({args.class2}): {results['scores'][1]:.2f}")
    
    return 0


if __name__ == "__main__":
    main()
