# Unordered list of interesting books and papers


## Books

1.  [Reinforcement Learning: An Introduction (Barto, Sutton)](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
  
## Papers

| Paper Link | Short Description |
|------------|-------------------|
| [Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://arxiv.org/pdf/2003.04960.pdf?fbclid=IwAR3n0MndHpbiWI1-Wfds5jTXSkwXwpo1mf7jaK-64J4heyYOnYO76qnEWCE) | Survey of curriculum learning papers. Formalises curriuclum learning based on a variety of attributes and gives a good introduction into the topic. |
| [Curriculum Design for Machine Learners in Sequential Decision Tasks](https://beipeng.github.io/files/2018ieee-tetci-peng.pdf) | Talks about curricula desgined by non-experts (i.e. people who do not know much/anything about a given domain). Uses "dog training" game as the basis of their experiments. Users can design a curriculum by sequencing any number of tasks in more or less complex environments. A trainer model is used to provide feedback to the agent and go through the curriculum. They measure how good a curriculum is by the number of feedbacks that the trainer has to give to the agent i.e. if a curriculum is well designed the agent will require a relatively smaller number of feedbacks from the trainer to move on to the next task. They use three different trainer behaviours and show that the type of the trainer oes not influence the impact of curriuclum design i.e. if a curriculum is well designed under one trainer it is also well designed under another trainer. Another condition for a curriculum to be considered good is that the number of feedbacks over the entire curriculum (with the target task) should be   Results show that non-experts can design a good curriculum.
