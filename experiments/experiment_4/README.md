# Experiment Purpose
The goal of the experiment is to investigate the impact of the time spent on an easier-level task on the training time of 
the agent in the entire curriculum. The curriculum is defined based on the stop condition set in a more challenging task, 
specifying a particular average evaluation score that the agent should achieve to complete learning. Additionally, the 
experiment aims to explore how the time spent on an easier-level task influences the final evaluation of the agent in a 
more challenging task, determined by setting a stop condition on a specific number of episodes and measuring the average 
evaluation of the agent after this learning time.

# Experiment Implementation
The experiment was conducted on the LavaCrossing environment with difficulty levels 0 and 1. To examine the impact of 
learning time at the easier level on the training time of the agent at the more challenging level, five different episode
values were tested as stop conditions: 750/1000/1250/1500/1750 in the easier task. The stop condition for the minimum 
average evaluation score of the agent in the second task was set to 0.8. For each specific episode value in the easier 
task, the experiment was repeated 10 times, changing the agent's seed (used for initializing the neural network architecture) 
to mitigate the influence of randomness and focus the results for greater reliability.

Moving on to the second part of the experiment, investigating the influence of learning time at the easier level on the 
final average evaluation of the agent at the more challenging level, the same five episode values were examined as stop 
conditions: 750/1000/1250/1500/1750 in the easier task. However, this time, the maximum number of episodes was set to 1500 
as the stop condition in the more challenging task. After the completion of learning, a 100-fold evaluation of the agent 
is performed to obtain a result as close as possible to the expected evaluation value. This approach allows comparing how 
the final average evaluation of the agent changed after the completed learning process in the curriculum on the final 
task with a more challenging level. Similarly, for each different number of episodes in the easier task, all curriculum 
calls were repeated 10 times.