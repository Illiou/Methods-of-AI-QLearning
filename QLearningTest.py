from MDP import MDP
from QLearning import QLearning
import Gridworld
import DefaultConstants as Default

gridworld_list = Gridworld.make_list_from_file("3by4.grid")

mdp = MDP(state_list=gridworld_list,
          field_rewards=Default.FIELD_REWARDS,
          obstacle_fields=Default.OBSTACLE_FIELDS,
          actions=Default.ACTIONS,
          transition_probabilities=Default.TRANSITION_PROBABILITIES)

qlearning = QLearning(env_perform_action=mdp.perform_action,
                      state_list=gridworld_list,
                      goal_fields=Default.GOAL_FIELDS,
                      obstacle_fields=Default.OBSTACLE_FIELDS,
                      actions=Default.ACTIONS,
                      discount_factor=Default.DISCOUNT_FACTOR,
                      learning_rate=0.1,
                      epsilon=0.5,
                      convergence_threshold=1000)

print("---Instance variables---")
print(qlearning.states)
print(qlearning.goal_states)
print(qlearning.actions)
print(qlearning.discount_factor)
print(qlearning.learning_rate)
print(qlearning.epsilon)
print(qlearning.q_function)
print()

"""current_state = (0, 2)
print("Current state:", current_state)
while current_state not in qlearning.goal_states:
    current_state = qlearning.q_learning_step(current_state, mdp.env_perform_action)
    print(*qlearning.format_q_function())
    print("Current state:", current_state)
current_state = qlearning.q_learning_step(current_state, mdp.env_perform_action)
print(*qlearning.format_q_function())
print("Current state:", current_state)"""

# qlearning.q_learning_episode(mdp.env_perform_action)
qlearning.q_learning_until_convergence()
print(*qlearning.format_q_function())
print("Nr. of iterations:", qlearning.last_convergence_episode_count)
Gridworld.print_policy(qlearning.format_policy())
