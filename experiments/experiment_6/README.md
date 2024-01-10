# Experiment Purpose

The aim of this experiment is to analyze whether skipping some of the tasks
during the curriculum can improve convergance.

The idea behind skipping is that the agent might learn just enough during the first
task to be able to complete the last task fast enough and that the time
spent in the intermediate tasks is wasted.

# Experiment Implementation

To achieve the experiment's goal, agents learning curves will be compared 
between two learning scenarios on LunarLander environmnent. Both scenarios aim to 
produce an agent that is able to achieve satisfactory* results on the 
LunarLander environment with difficulty level set to 4.

In both scenarios a Curriculum Learning approach is used. An agent starts 
on the easiest level (difficulty level 0). It moves to the next one once it 
reaches an evaluation score of 200. The run ends once the agent reaches that 
threshold on the final level (difficulty level 4).

However in the first scenario the agent will only train on the task with difficulty 0
and then immediately skip to the task with difficulty 4. In the second scenario
the agent also train on the dask with difficulty 3 after finishing the task with difficulty 0
and before starting the task with difficulty 4.

This experiment will be conducted using DQN. 
Each run will be repeated 10 times to minimize the impact of randomness on the results.

**satisfactory i.e. reaching the evaluation score threshold of 200* 