"""
This class can be initialized with a gridworld and some parameters and
is then able to find an optimal policy with Q-learning.

The assignment document talks about checking for convergence by comparing
last episode's policy to the current one. If you stop as soon as the
policy doesn't change in one episode you get pretty bad results, therefore
we included a convergence threshold, which is the number of episodes
in which the policy didn't change, after which the policy will be considered
to have converged.
"""

import random


class QLearning:
    def __init__(self, env_perform_action, state_list, goal_fields, obstacle_fields, actions, discount_factor,
                 learning_rate, epsilon, convergence_threshold, decimal_places=5):
        """
        Sets up a representation of the gridworld given the following parameters.
        "Field" here refers to the letters or signs with which different states are represented.

        :param env_perform_action: function of the environment which gives back tuple (reward, follow-up state)
                                   given a state and an action
        :param state_list: two-dimensional list of possible states represented as specific fields
        :param goal_fields: list of fields which are considered terminal states
        :param obstacle_fields: list of fields which are considered obstacles
        :param actions: list of possible movements in tuple notation, i.e. (x_coordinate_offset, y_coordinate_offset)
        :param discount_factor: float being the discount factor gamma
        :param learning_rate: float being the learning rate alpha
        :param epsilon: float being epsilon for the epsilon-soft policy
        :param convergence_threshold: number of episodes without change of the policy for which
                                      it will be considered to have converged
        :param decimal_places: optionally change number of decimal places numbers are rounded to
        """

        # list of reachable states in (x, y) coordinate tuple notation
        # obstacles are left out as they are not reachable by an agent
        self.states = []
        # list of the states considered terminal states
        self.goal_states = []
        # filling the two above
        for y, line in enumerate(state_list):
            for x, field in enumerate(line):
                if field not in obstacle_fields:
                    self.states.append((x, y))
                if field in goal_fields:
                    self.goal_states.append((x, y))

        # making sure goal_states is a subset of states as they are unreachable otherwise
        if not set(self.goal_states).issubset(self.states):
            raise Exception("Goal states cannot be obstacles")

        # save as instance variables
        self.env_perform_action = env_perform_action
        self.actions = actions
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.convergence_threshold = convergence_threshold
        self.decimal_places = decimal_places

        # dimensions of the Gridworld for formatting in the end
        self.dim = (len(state_list[0]), len(state_list))
        # initialize action-value function Q
        self.reset_q_function()
        # initialize policy
        self.update_policy()
        # initialize current state starting with a random state
        self.reset_current_state()

        # for saving the number of Q-learning episodes it took for the policy to converge
        self.last_convergence_episode_count = -1


    def reset_q_function(self):
        """(Re)sets action-value function Q to 0"""
        self.q_function = {(s, a): 0 for s in self.states for a in self.actions}


    def reset_current_state(self):
        """Sets current state to random (starting) state"""
        self.current_state = random.choice(self.states)


    def update_policy(self):
        """Updates policy based on current action-value function Q"""
        self.policy = {s: max(self.actions, key=lambda a: self.q_function[s, a]) for s in self.states}


    def q_learning_step(self):
        """
        Performs one Q-learning step, i.e. the agent moves one step in the Gridworld
        and the Q-values are updated according to the feedback from the environment.
        """

        s = self.current_state  # for readability
        # epsilon-soft policy: choose random action (including greedy action) with probability epsilon
        if random.random() < self.epsilon:
            a = random.choice(self.actions)
        # with probability 1 - epsilon choose greedy action
        else:
            a = max(self.actions, key=lambda a_: self.q_function[s, a_])
        # perform action and observe reward and follow-up state from environment
        r, s_prime = self.env_perform_action(s, a)
        # perform q_function update
        if s not in self.goal_states:
            greedy_q = max(self.q_function[s_prime, a_prime] for a_prime in self.actions)
        else:
            greedy_q = 0  # if current state is a goal state future reward will always be 0
        updated_q = self.q_function[s, a] + self.learning_rate * (r + self.discount_factor * greedy_q - self.q_function[s, a])
        self.q_function[s, a] = round(updated_q, self.decimal_places)
        # set new current state
        self.current_state = s_prime


    def q_learning_episode(self):
        """
        Performs Q-learning steps until an action is performed in a terminal state and then updates the policy.
        """
        self.reset_current_state()
        while self.current_state not in self.goal_states:
            self.q_learning_step()
        self.q_learning_step()  # necessary to observe reward from goal state
        self.update_policy()


    def q_learning_until_convergence(self):
        """
        Performs Q-learning episodes until the policy hasn't changed
        for a given number of episodes (i.e. it has converged)
        """

        self.last_convergence_episode_count = 0
        policy_unchanged_count = 0
        while policy_unchanged_count < self.convergence_threshold:
            old_policy = self.policy.copy()
            self.q_learning_episode()
            self.last_convergence_episode_count += 1
            if old_policy == self.policy:
                policy_unchanged_count += 1
            else:
                policy_unchanged_count = 0


    def format_q_function(self):
        """
        :return: nested list with dimensions of the original Gridworld containing for each state
                 tuples with action-value function Q values for each action
        """
        def all_q_values(s):
            return tuple(self.q_function.get((s, a), 0) for a in self.actions)

        return [[all_q_values((x, y)) for x in range(self.dim[0])] for y in range(self.dim[1])]


    def format_policy(self):
        """
        :return: nested list with dimensions of the original Gridworld
                 containing actions according to best q value of each state
        """
        return [[self.policy.get((x, y), (0, 0)) for x in range(self.dim[0])] for y in range(self.dim[1])]
