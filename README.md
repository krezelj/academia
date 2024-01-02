# academia

This package’s purpose is to provide easy-to-use tools for Curriculum Learning.
It is a part of an engineering thesis at Warsaw University of Technology that 
touches on the topic of curriculum learning.

## Documentation

https://academia.readthedocs.io/

## Sources
An unordered list of interesting books and papers

### Books

1.  [Reinforcement Learning: An Introduction (Barto, Sutton)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
  
### Papers

| Paper Link | Short Description |Related Papers |
|------------|-------------------| - |
| [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://arxiv.org/pdf/2003.04960.pdf?fbclid=IwAR3n0MndHpbiWI1-Wfds5jTXSkwXwpo1mf7jaK-64J4heyYOnYO76qnEWCE) | Survey of curriculum learning papers. Formalises curriuclum learning based on a variety of attributes and gives a good introduction into the topic. ||
| [Curriculum Design for Machine Learners in Sequential Decision Tasks](https://beipeng.github.io/files/2018ieee-tetci-peng.pdf) | Talks about curricula desgined by non-experts (i.e. people who do not know much/anything about a given domain). Uses "dog training" game as the basis of their experiments. Users can design a curriculum by sequencing any number of tasks in more or less complex environments. A target (more difficult) task is also provided to the user but they are not allowed to include it in the curriculum. A trainer model is used to go through the curriculum and provide feedback to the agent. They measure how good a curriculum is by the number of feedbacks that the trainer has to give to the agent i.e. if a curriculum is well designed the agent will require a relatively smaller number of feedbacks from the trainer to move on to the next task. They use three different trainer behaviours and show that the type of the trainer oes not influence the impact of curriuclum design i.e. if a curriculum is well designed under one trainer it is also well designed under another trainer. Another condition for a curriculum to be considered good is that the number of feedbacks over the entire curriculum (with the target task) should be smaller than the number of feedbacks when training on the target task alone. Results show that non-experts can design a better-than-random curriculum when it comes to reducing number of feedbacks on the target task alone but are not better-than-random in desigining a curriculum that decreases the overall number of feedbacks. | [Language and Policy Learning from Human-delivered Feedback](https://beipeng.github.io/files/2015icra-peng.pdf),<br/><br/> [Learning behaviors via human-delivered discrete feedback](https://link.springer.com/article/10.1007/s10458-015-9283-7)
| [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) | Not directly related to Curriculum Learning but related to Reinforcement Learning. | 
|[A Deep Hierarchical Approach to Lifelong Learning in Minecraft](https://arxiv.org/pdf/1604.07255.pdf)|Haven't read it yet, not directly connected to CL but should still be helpful||
