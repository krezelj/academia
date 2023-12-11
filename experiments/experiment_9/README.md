# Experiment Purpose

This is the second one of two experiments which aim is to analyze whether 
skipping level(s) in curriculum affects algorithm's performance on the 
highest level. 

It is possible that an algorithm does not need to go through every single 
difficulty level to be able to learn quickly on the highest level. It might be
enough to learn the basics of the game on the easiest levels and then go
straight into the highest level without affecting training times there. This
theory will be tested in this experiment.

# Experiment Implementation

To achieve the experiment's goal, agents learning curves will be compared 
between two learning scenarios on DoorKey environmnent. Both scenarios aim to 
produce an agent that is able to achieve satisfactory* results on the 
DoorKey environment with difficulty level set to 2.

In the first scenario an agent has to go though every difficulty level up to
level 2, starting on level 0 (0 -> 1 -> 2).

In the second scenario agent also starts on level 0, hovewer once it reaches 
satisfactory results there it skips level 1 and moves straight into level 2 
(0 -> 2).

This experiment will be conducted using PPO algorithm. Each run will be repeated 
10 times to minimize the impact of randomness on the results.

**satisfactory i.e. reaching the evaluation score threshold of 0.9* 
