# Experiment Purpose

The aim of this experiment is to analyze the impact of Curriculum Learning approach on
the training time of the agent. 

In theory, an agent should be able to achieve good results on an environment
quicker if it first gets trained on easier versions of the environment.

# Experiment Implementation

To achieve the experiment's goal, agents learning curves will be compared 
between two learning scenarios on MsPacman environmnent.

In the first scenario a Curriculum Learning approach is used. An agent starts 
on the easiest level (difficulty level 0). After a certain number of steps it
moves to the next level (difficulty level 1) where it again trains for
a certain number of steps. Finally it moves to the final level (difficulty level 3)

The second scenario utilizes a more traditional approach. An agent doesn't 
move between different difficulty levels - instead it goes straight into the 
final level and tries to learn it from scratch. The agent spends the same number
of steps training on that level as it did in all the curriculum tasks combined.