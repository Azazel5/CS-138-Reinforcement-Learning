# CS-138-Reinforcement-Learning

All assignments and reports for CS 138 at Tufts University

The subfolders will be:


1. Programming Assignment #1
2. Programming Assignment #2
3. Programming Assignment #3
4. Programming Assignment #4


## Programming Assignment #1

For this assignment, we were expected to write a program which solves and adds onto one of the two programming
exercises listed in Sutton and Barto chapter 2. I decided to work with 10-armed non-stationary bandit problem
i.e. the rewards are not discretely defined. 

To run the environment, create a virtual environment and installing the required packages using pip.

~~~
python -m venv env 
pip install -r requirements.txt
python rl.py
~~~

And running this block will define the environment, agent, and run the experiment 2000 times for 10000 timesteps each.
One novel idea in this experiment is to compare two different styles of updating the action's reward estimates, using
a defined alpha value or taking the average of all actions at any time step. 

A successful run of the program will generate two graphs which shows us a direct comparative example of what strategy
yeilds higher rewards and which give us higher chance of optimal action selection.
