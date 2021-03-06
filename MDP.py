"""
This MDP class provides the environment for Q-learning.
It's very similar to the MDP in first programming task, but stripped of everything except
for the init function and it now has method that gives information about performing an action
in a specific state.

The initialization is mostly the same as in the QLearning class, but to keep
Q-learning and environment independent this is somewhat necessary.
"""

import random


class MDP:
    def __init__(self, state_list, field_rewards, obstacle_fields, actions, transition_probabilities):
        """
        Initializes an MDP with the following parameters.
        "Field" here refers to the letters or signs with which different states are represented.

        :param state_list: two-dimensional list of possible states represented as specific fields
        :param field_rewards: dictionary which maps fields in state_list to a reward value
        :param obstacle_fields: list of fields which are considered obstacles
        :param actions: list of possible movements in tuple notation, i.e. (x_coordinate_offset, y_coordinate_offset)
        :param transition_probabilities: dictionary of transition probabilities,
                                         mapping a probability to "straight" and "lateral" movement
        """

        # list of reachable states in (x, y) coordinate tuple notation
        # obstacles are left out as they are not reachable by an agent
        self.states = []
        # function implemented as dictionary which returns the immediate reward for the given state
        # as noted above it is only dependant on the state, not the action and therefore only called with the former
        self.rewards = {}
        # filling the three above
        for y, line in enumerate(state_list):
            for x, field in enumerate(line):
                if field not in obstacle_fields:
                    self.states.append((x, y))
                    self.rewards[(x, y)] = field_rewards[field]

        # save as instance variables
        self.actions = actions
        self.transition_probabilities = transition_probabilities


    def perform_action(self, s, a):
        """
        Performs the given action in the given state and returns the immediate reward and the next state.
        :param s: current state
        :param a: action to be performed
        :return: tuple of immediate reward, follow-up state
        """

        if s not in self.states:
            raise Exception("Invalid state given.")
        if a not in self.actions:
            raise Exception("Invalid action given")

        # with probability (1 - probability of going straight) set action to random orthogonal action
        if random.random() >= self.transition_probabilities["straight"]:
            a = random.choice([(a[1], a[0]), (-a[1], -a[0])])
        # calculate follow-up state
        s_prime = (s[0] + a[0], s[1] + a[1])
        # if follow-up state not a valid state, i.e. a wall, stay in current state
        if s_prime not in self.states:
            s_prime = s
        return self.rewards[s], s_prime
