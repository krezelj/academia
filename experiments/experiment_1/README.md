# Experiment Purpose

The goal of the experiment is to analyze the impact of the epsilon value, to which the DQN agent is reset after completing a LearningTask, on its evaluation in subsequent LearningTasks with progressively higher difficulty levels.

We aim to find the optimal value from the tested epsilon values to see which reset value yields the best results. The objective is for the agent not to forget previously learned behaviors and make the most of previously acquired tactics.

This is particularly useful in Curriculum Learning, where the idea is to train the agent on simpler environments so that it develops certain behaviors that can be transferred to more challenging tasks. Therefore, selecting the epsilon value in the DQN algorithm, which follows the epsilon-greedy strategy, is crucial. It is important that after completing a task with the easiest level, the agent makes more of its own choices than following random actions. It's also crucial to balance epsilon, as setting it too low immediately would lead the agent to exploit rather than explore, hindering further learning.

# Experiment Implementation

To conduct the mentioned experiment, five different experiments will be carried out, each with a different epsilon reset value after completing each task. Additionally, each experiment will have a properly selected epsilon decay to prevent epsilon from decreasing too quickly to values less than one-hundredth of a percent. The tested values are as follows:

| Epsilon Reset Value | Epsilon Decay |
| ------------------- | ------------- |
| 1.0                 | 0.999         |
| 0.6                 | 0.9994        |
| 0.3                 | 0.9995        |
| 0.1                 | 0.9996        |
| 0.03                | 0.9997        |

Each experiment will be repeated `10 times`, and the results will be averaged to obtain more stable outcomes and minimize the hypothesis that the obtained results are purely coincidental.

The experiment will be conducted on the LavaCrossing environment at difficulty levels 0, 1, 2, with the agent's learning stopping condition set at `min_evaluation_score` = 0.8.
