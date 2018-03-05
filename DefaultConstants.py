"""
This files contains values used throughout the Gridworld, MDP and Q-learning programs
to be able to change them in a single place.
"""

# default values for MDP and Q-learning
FIELD_REWARDS = {"F": -0.04, "E": 1, "P": -1}
GOAL_FIELDS = ["E", "P"]
OBSTACLE_FIELDS = ["O"]
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
TRANSITION_PROBABILITIES = {"straight": 0.8, "lateral": 0.1}

# default values for Q-learning
DISCOUNT_FACTOR = 1.0
LEARNING_RATE = 0.1
EPSILON = 0.5
CONVERGENCE_THRESHOLD = 100

# default values for pretty printing
FIELD_MAPPING = {"F": " ", "O": "■", "E": "+", "P": "-"}
BOUNDARY_CHAR = "█"
ACTION_MAPPING = {(0, -1): "↑", (1, 0): "→", (0, 1): "↓", (-1, 0): "←", (0, 0): " "}
OUTPUT_WIDTH = 90

# other
SLEEP_TIME = 1.5
YES_SYNONYMS = ["y", "yes", "ye", "yea", "yeah", "yep", "yay", "aye", "arr", "sure",
                "j", "ja", "jap", "jep", "jo", "joa"]
NO_SYNONYMS = ["n", "no", "nay", "nope", "exit", "quit", "stop", "nein", "ne", "nö"]
