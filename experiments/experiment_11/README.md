# Experiment Purpose

The aim of this experiment is to analyze whether a specific (0.6) epsilon reset value
controlling the exploration of a DQN agent leads to better results for a curriculum.

# Experiment Implementation

This experiment is identical to the first scenario described in "experiment 6".
The only difference is that the DQN agent now resets to epsilon value equal to 0.6 (instead of 0.3)
after the first task and epsilon decay value is changed to 0.9994 (instead of 0.9995).