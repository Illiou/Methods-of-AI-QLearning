# Gridworld Q-learning
This Python script reads a text file containing a Gridworld and then performs
Q-learning on it to find an action policy that maximizes the reward.

The format of the Gridworld file should be similar to this example:

    F F F E
    F O F P
    F F F F

With
* `F` being an empty cell
* `O` being an obstacle
* `E` being a goal state/terminal state with positive reward
* `P` being a goal state/terminal state with negative reward

**See the files for more details on the implementation.**

### Calling the script
To call this script just enter `python Gridworld.py` into the console
(assuming you have Python installed of course). All necessary parameters
will be asked for in the program.  
If you want to call the script with an input Gridworld file directly,
you can do so by calling `python Gridworld.py yourgridworld.grid`.

The script was tested in Python 3.6. No guarantees that it will work in older versions.  
The main program is `Gridworld.py` which uses the other files.

### Known issues (of PyCharm...)
(Leaving this in here even though I mysteriously didn't have this problem this time...)
* In case you are using PyCharm:  
The "Press Enter to continue" doesn't work properly within the PyCharm console for some reason.  
Workaround: enter some character before pressing Enter.
