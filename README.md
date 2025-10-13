# CS-138-Reinforcement-Learning

All assignments and reports for CS 138 at Tufts University

The subfolders will be:


1. Programming Assignment #1
2. Programming Assignment #2
3. Programming Assignment #3
4. Programming Assignment #4


## Programming Assignment #1

For this assignment, we were expected to write a program which solves and adds onto one of the two programming
exercises listed in Sutton and Barto chapter two, question 2.5. I decided to work with 10-armed non-stationary bandit problem
i.e. the rewards are not discretely defined. 

To run the environment, create a virtual environment and installing the required packages using pip.

~~~
python -m venv env 
source env/bin/activate
pip install -r requirements.txt
python rl.py
~~~

And running this block will define the environment, agent, and run the experiment 2000 times for 10000 timesteps each. We compare two different styles of updating the action's reward estimates, using
a defined alpha value or taking the average of all actions at any time step. One novel idea, done 200 times for 1000 time steps each (due to time constraints), is add a volatile environment to see how alpha values need to be changed in such scenarios.

A successful run of the program will generate three graphs which shows us a direct comparative example of what strategy
yeilds higher rewards and which give us higher chance of optimal action selection. 

## Programming Assignment #2

In this assignment, we created an agent that was capable of utilizing Monte Carlo methods to learn a policy that lets it go from start to finsih on an imaginary racetrack. This is Sutton and Barto programming exercise 5.12. For this, I have used a numpy matrix to represent the racetrack positions and a python dictionary
to translate strings to positional representations i.e. '#' is a wall, 'S' is the starting position, etc.

The code can be run by: 

1. Cloning the repository
2. Navigating to the Programming Assignment 2 folder
3. Creating a python virtual environment and activating it
4. Installing the reqiorements
5. Running the python file

~~~
git clone https://github.com/Azazel5/CS-138-Reinforcement-Learning.git
cd Programming Assignment 2
python -m venv env 
source env/bin/activate
pip install -r requirements.txt
python rl.py
~~~