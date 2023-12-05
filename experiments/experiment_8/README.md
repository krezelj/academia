# Experiment Purpose

The aim of this experiment is to analyze the influence of reward density on the
process of Curriculum Learning.

Curriculum Learning is one way of addressing the problem of sparse reward
in environments so it is expected that in environments with dense rewards
Curriculum Learning will have a less significant impact.

# Experiment Implementation

To achieve the experiment's goal, agent learning curves using Curriculum Learning
and No Curriculum will be compared between two environment configuarations.

In the first configuration the reward will be sparse and given only after the agent
reaches the goal. In the second configuration the reward will be more dense
and given for completing subtasks that enable the agent to reach the goal.

To conduct this experiment we will use our custom `BridgeBuilding` environment
which is designed with both sparse and dense reward in mind. This allows 
for a more objective comparison since the underlying task will not change.
The agent will still have to perform the same sequence of actions to reach 
the goal.

`BridgeBuilding` environment consists of a river with two banks. The task
is for the agent to build a bridge across the river using boulders scattered
next to the agent on the left bank. 

In the sparse configurations the agent will only obtain the reward upon reaching 
the right bank and the reward will be inversly proportional to the time spent 
while building the bridge.

In the dense configurations the agent additionally obtains rewards or penalties when
making the bridge longer or shorter respectively.

In both configurations and both curriculum and non-curriculum approach 
the satisfactory episode reward is set to `1.8` which (combined with `max_steps`
in the environment set to `500`) is equivalent of the agent reaching the goal
within 100 time steps and the bridge being fully built.