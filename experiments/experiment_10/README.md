# Experiment Purpose
*The aim of the experiment is exactly the same as in `experiment number 3`. The 
experiment has been repeated with the curriculum expanded by one additional task 
with a `difficulty` level equal to `4`. This is the maximum recommended level as indicated 
in the Lunar Lander environment documentation.*

The aim of this experiment is to analyze the impact of Curriculum Learning approach on
the training time of the agent. 

In theory, an agent should be able to achieve good results on an environment
quicker if it first gets trained on easier versions of the environment.

# Experiment Implementation

To achieve the experiment's goal, agents learning curves will be compared 
between two learning scenarios on LunarLander environmnent. Both scenarios aim to 
produce an agent that is able to achieve satisfactory* results on the 
LunarLander environment with difficulty level set to 3.

In the first scenario a Curriculum Learning approach is used. An agent starts 
on the easiest level (difficulty level 0). It moves to the next one once it 
reaches an evaluation score of 200. The run ends once the agent reaches that 
threshold on the final level (difficulty level 3).

The second scenario utilizes a more traditional approach. An agent doesn't 
move between different difficulty levels - instead it goes straight into the 
final level and tries to learn it from scratch. The run ends once the agent 
reaches evaluation score threshold of 200 or once the maximum number of 
steps is reached.

This experiment will be conducted on both DQN and PPO algorithms. Each run 
will be repeated 10 times to minimize the impact of randomness on the results.

**satisfactory i.e. reaching the evaluation score threshold of 200* 