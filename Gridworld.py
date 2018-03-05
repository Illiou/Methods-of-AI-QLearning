import os
import sys
import time
import threading
from MDP import MDP
from QLearning import QLearning
import DefaultConstants as Default


def init():
    """
    Ask for a Gridworld file and initializes an MDP as environment and Q-learning object with it, then calls the menu.
    """
    print_headline("Gridworld Selection")
    gridworld = read_gridworld_file()

    environment = MDP(state_list=gridworld,
                      field_rewards=Default.FIELD_REWARDS,
                      obstacle_fields=Default.OBSTACLE_FIELDS,
                      actions=Default.ACTIONS,
                      transition_probabilities=Default.TRANSITION_PROBABILITIES)

    q_learning = QLearning(env_perform_action=environment.perform_action,
                           state_list=gridworld,
                           goal_fields=Default.GOAL_FIELDS,
                           obstacle_fields=Default.OBSTACLE_FIELDS,
                           actions=Default.ACTIONS,
                           discount_factor=Default.DISCOUNT_FACTOR,
                           learning_rate=Default.LEARNING_RATE,
                           epsilon=Default.EPSILON,
                           convergence_threshold=Default.CONVERGENCE_THRESHOLD)

    print("Your input Gridworld:")
    print_gridworld(gridworld)

    while show_menu(q_learning):
        pass

    print_headline("See you later")


def show_menu(q_learning):
    """
    Shows a menu and calls the appropriate functions based on what is selected.
    :param q_learning: QLearning object to work with
    :return: True if menu needs be shown again, False otherwise
    """
    print_headline("Menu")
    print("What do you want to do? Enter the corresponding number.\n")
    print("[1] Automatic Q-learning until convergence")
    print("[2] Automatic Q-learning episode")
    print("[3] Manual step-by-step Q-learning episode")
    print("[4] Print current Q-function values and derived policy")
    print("[5] Reset Q-function and derived policy")
    print("[6] Change the exploration rate for the epsilon-soft policy (epsilon). Currently set to {}".format(q_learning.epsilon))
    print("[7] Change the learning rate (alpha). Currently set to {}".format(q_learning.learning_rate))
    print("[8] Change the discount factor of future rewards (gamma). Currently set to {}".format(q_learning.discount_factor))
    print("[9] Change the convergence threshold (episodes with unchanged policy). Currently set to {}".format(q_learning.convergence_threshold))
    print("[0] Exit the program\n")
    chosen_item = secure_input(int, text="Choose: ", lower_bound=0, upper_bound=9)

    # automatic Q-learning until convergence
    if chosen_item == 1:
        print_sep()
        automatic_q_learning_until_convergence(q_learning)
        input("Press Enter to return to the main menu...")
        return True
    # automatic Q-learning episode
    elif chosen_item == 2:
        while True:
            automatic_q_learning_episode(q_learning)
            if not ask_yes_no("Do you want to run another episode? (No will return to the main menu)\n> "):
                break
        return True
    # manual Q-learning episode
    elif chosen_item == 3:
        manual_q_learning_episode(q_learning)
        return True
    # print Q-function and policy
    elif chosen_item == 4:
        print_sep()
        print_q_function_and_policy(q_learning)
        input("\nPress Enter to return to the main menu...")
        return True
    # reset Q-function
    elif chosen_item == 5:
        q_learning.reset_q_function()
        q_learning.update_policy()
        print("\nQ-function successfully reset.")
        time.sleep(Default.SLEEP_TIME)
        return True
    # change epsilon
    elif chosen_item == 6:
        print_sep()
        q_learning.epsilon = secure_input(float, text="Enter a new epsilon value between 0 and 1: ",
                                          lower_bound=0, upper_bound=1)
        print("\nEpsilon value successfully changed.")
        time.sleep(Default.SLEEP_TIME)
        return True
    # change learning rate
    elif chosen_item == 7:
        print_sep()
        q_learning.learning_rate = secure_input(float, text="Enter a new learning rate between 0 and 1: ",
                                                lower_bound=0, upper_bound=1)
        print("\nLearning rate successfully changed.")
        time.sleep(Default.SLEEP_TIME)
        return True
    # change discount factor
    elif chosen_item == 8:
        print_sep()
        q_learning.discount_factor = secure_input(float, text="Enter a new discount factor between 0 and 1: ",
                                                  lower_bound=0, upper_bound=1)
        print("\nDiscount factor successfully changed.")
        time.sleep(Default.SLEEP_TIME)
        return True
    # change convergence threshold
    elif chosen_item == 9:
        print_sep()
        q_learning.convergence_threshold = secure_input(int, text="Enter a new convergence threshold: ",
                                                        lower_bound=1)
        print("\nConvergence threshold successfully changed.")
        time.sleep(Default.SLEEP_TIME)
        return True
    # exit program
    else:
        return False


def automatic_q_learning_until_convergence(q_learning):
    """
    Performs Q-learning episodes until the policy hasn't changed for a given number of episodes,
    then prints the results.
    A new thread is used to perform Q-learning in order to stay responsive during calculation.
    :param q_learning: QLearning object to work with
    """

    # fancy threading stuff to give feedback during the calculation so the user knows it hasn't crashed yet
    q_learning_thread = threading.Thread(target=q_learning.q_learning_until_convergence, daemon=True)
    q_learning_thread.start()
    print("\nCalculating", end="", flush=True)
    # check every second if Q-learning thread has finished, if not print a dot
    while q_learning_thread.is_alive():
        time.sleep(1)
        print(".", end="", flush=True)
    print("\n\nCalculated {} episodes, policy converged after {} episodes.".format(
          q_learning.last_convergence_episode_count,
          q_learning.last_convergence_episode_count - q_learning.convergence_threshold))
    time.sleep(Default.SLEEP_TIME)
    print_headline("Results")
    print_q_function_and_policy(q_learning)


def automatic_q_learning_episode(q_learning):
    """
    Performs one Q-learning episode, then prints the results.
    :param q_learning: QLearning object to work with
    """
    q_learning.q_learning_episode()
    print_headline("Results")
    print_q_function_and_policy(q_learning)


def manual_q_learning_episode(q_learning):
    """
    Performs one Q-learning step, then prints the Q-function values.
    :param q_learning: QLearning object to work with
    """
    q_learning.reset_current_state()
    while True:
        last_state = q_learning.current_state
        print_sep()
        print("Agent was previously on field: {}".format(q_learning.current_state))
        q_learning.q_learning_step()
        print("Agent moved to field: {}".format(q_learning.current_state))
        print("\n\nCalculated Q-function values:")
        print_q_function(q_learning.format_q_function())
        if last_state in q_learning.goal_states:
            print("Agent moved from a terminal state and therefore the episode ended.\n")
            input("Press Enter to return to the main menu...")
            break
        if not ask_yes_no("Run next step? (No will return to the main menu)\n> "):
            break
    q_learning.update_policy()


def print_q_function_and_policy(q_learning):
    """
    Prints current Q-function values and derived policy.
    :param q_learning: a QLearning object
    """
    print("Calculated Q-function values:")
    print_q_function(q_learning.format_q_function())
    print("Derived policy:")
    print_policy(q_learning.format_policy())


def read_gridworld_file():
    """
    Gets Gridworld filename from starting parameters or the user,
    the returns nested list of it's content.
    :return: nested list of characters
    """
    # check if Gridworld filename was given when starting program
    try:
        return make_list_from_file(sys.argv[1])
    # if not ask user instead
    except IndexError:
        while True:
            path_to_file = secure_input(str, text="Enter Gridworld filename (e.g. 3by4.grid): ")
            if not os.path.isfile(path_to_file):
                print("Invalid filename, try again!\n")
                continue
            print()
            return make_list_from_file(path_to_file)


def make_list_from_file(file):
    """
    Opens the given file and returns the characters, stripped from whitespace, in nested list form.
    :param file: name of the file to be opened
    :return: nested list of characters
    """
    with open(file) as f:
        lines = [[char for char in line if char != "\n" and char != " "] for line in f]
    return lines


def ask_yes_no(text):
    """
    Asks for yes or no with the passed text.
    :param text: text to show with the input command
    :return: True if yes was input, False if no
    """
    print_sep(spacing_top=False)
    while True:
        result = input(text).strip().lower()  # strip whitespace and make lowercase
        if result in Default.YES_SYNONYMS:
            return True
        if result in Default.NO_SYNONYMS:
            return False
        print("What d'ye want, matey?\n")


def secure_input(input_type, text="", lower_bound=None, upper_bound=None):
    """
    Asks the user for an input of a given type and range until appropriate
    input is given and then returns it.
    :param input_type: type conversion function matching the expected input type
    :param text: text to show with the input command
    :param lower_bound: lower bound of accepted range
    :param upper_bound: upper bound of accepted range
    :return: valid given input
    """
    while True:
        try:
            result = input_type(input(text))
            if lower_bound is not None and result < lower_bound:
                raise ValueError
            if upper_bound is not None and result > upper_bound:
                raise ValueError
            return result
        except ValueError:
            print("Invalid input. Try again!\n")


def print_headline(text, width=Default.OUTPUT_WIDTH):
    """
    Prints a nicely formatted headline with passed text.
    :param text: text in the headline
    :param width: optionally different width
    """
    print_sep(width, spacing_bottom=False)
    print("|{:^{w}}|".format(text, w=width))  # ^ centers the word between the filling spaces
    print_sep(width, spacing_top=False)


def print_sep(width=Default.OUTPUT_WIDTH, spacing_top=True, spacing_bottom=True):
    """
    Prints a line and spaces above and below.
    :param width: optionally different width
    :param spacing_top: pass False if no space above is needed
    :param spacing_bottom: pass False if no space below is needed
    """
    if spacing_top:
        print()
    # {:c<number} prints character "c" for "number" times
    print("#{:-<{w}}#".format("", w=width))
    if spacing_bottom:
        print()


def print_gridworld(gridworld, field_mapping=Default.FIELD_MAPPING, boundary_char=Default.BOUNDARY_CHAR):
    """
    Prints the Gridworld in a readable way with more distinct characters
    :param gridworld: nested list being the Gridworld
    :param field_mapping: dictionary mapping field characters to more distinct characters
    :param boundary_char: character for the boundary of the Gridworld
    """
    print()
    # prints boundary chars for the full length of the gridworld
    # 2 * because of the spaces in between, + 2 because of the side boundaries
    print(boundary_char * (2 * len(gridworld[0]) + 2))
    for line in gridworld:
        print(boundary_char, end="")
        for char in line:
            print(field_mapping[char], end=" ")
        print(boundary_char)
    print(boundary_char * (2 * len(gridworld[0]) + 2))
    print()


def print_policy(policy_list, action_mapping=Default.ACTION_MAPPING):
    """
    Prints the actions according to the policy in a visual way
    :param policy_list: nested list of actions according to policy
    :param action_mapping: dictionary mapping actions to readable characters like arrows
    """
    print()
    for line in policy_list:
        print(" ", end="")
        for char in line:
            print(action_mapping[char], end=" ")
        print()
    print()


def print_q_function(q_value_tuple_list, number_padding=5):
    """
    Prints the Q-function values with appropriate spaces and alignment
    :param q_value_tuple_list: nested list of Q-function value tuples
    :param number_padding: optionally specify different padding between numbers
    """
    if number_padding < 5:
        number_padding = 5
    # add maximal number of places before the decimal point to total number padding
    number_padding += len(str(round(max(max(max(q_value_tuple_list))))))
    # width of the whole gridworld
    total_width = len(q_value_tuple_list[0]) * (2 * number_padding + 4) + 1

    print("\n{:-<{w}}".format("", w=total_width))
    for line in q_value_tuple_list:
        # up values line
        print("|", end="")
        for tuple_ in line:
            print("  {:^{p}.3f} |".format(tuple_[0], p=2 * number_padding), end="")
        # left and right values line
        print("\n|", end="")
        for tuple_ in line:
            print("{:{p}.3f} | {:<{p}.3f}|".format(tuple_[3], tuple_[1], p=number_padding), end="")
        # down values line
        print("\n|", end="")
        for tuple_ in line:
            print("  {:^{p}.3f} |".format(tuple_[2], p=2 * number_padding), end="")
        # separation line
        print("\n{:-<{w}}".format("", w=total_width))
    print()


# only run if not imported from other file
if __name__ == "__main__":
    init()
