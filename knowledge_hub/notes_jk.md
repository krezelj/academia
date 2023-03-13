# Notes
**Jan Krężel**

## Performance Measures

The performance of curriculum learning can be measured in three different ways

- the total time it takes to design and execute a curriculum as well as learng the target task should be less than the time it takes to learn the target task alone
    - this is the strongest performance measure
    - it can be difficult to measure the time it took to design a curriculum, especially if it's designed by humans
- the time it takes to design a curriculum is treaded as a sunk cost. Instead only the time it takes to execute the curriculum and learn the target task should be less than the time it takes to learn the target task alone
    - this is also a strong measure but it's much more realistic to use
    - Depending on the task, I think we should use this measure in our research
- only the time it takes to learn the target  task, with or without the currriculum, is compared. The rest is ignored
    - this is the weakest measure
    - can possibly be used when the agent that passed the curriculum can be reused for many different tasks which may not be known at the time of executing the curriculum


---

## Sequencing

**NOTE** This is *my* understanding of sequencing, it could be incorrect.

A curriculum is a sequence (or more generally an acyclic, directed graph with a source and sink node) of samples and/or tasks.

The sequence can be made of different samples in a single task or different tasks that somehow build up to the target task.

For example, consider a self-driving car agent. A target task could be defined as "Go from point A to point B". 

A curriculum that is a sequence of samples from the target task could look like this
1. Put the agent on a straight road where A and B are its ends
1. Put the agent on a curved road ...
1. Put the agent on a road with a crossing
1. Put the agent on a road with a traffic-light controlled crossing
1. ...
1. Put the agent in a small city with other agents, traffic lights, crossings, etc.

A curriculum that is a sequence of tasks could look like this
1. Task the agent with driving in its lane on a straight road
1. Task the agent with stopping on red lights
1. Task the agent with obeying crossing laws
1. Task the agent with driving withing a speed limit but not too slow
1. ...
1. Target task same as before

Often sequencing targets can be more useful but it can be difficult to collect all that knowledge in one agent, especially if the tasks differ too much. For example we cannot simply train the agent on one task, then train it on another task and expect it to perform the first task equally well as it did before training on the second task, i.e. the knowledge can be lost between tasks and it's difficult to retain it in.  

---

## Connection to Active Learning

One way to design a curriculum is to create it online (while executing it instead of having it prepared beforehand). From what I understand there is often a concept of *Trainer/Teacher* when it comes to curriculum learning. It's another model that learns along the primary agent and tries to challange it with different tasks/samples.

One way a trainer could work is to receive some kind of feedback from the agent (or assess the agent by itself) and figure out the tasks with which the agent still has difficulties. Then it can generate tasks that are simpler versions of the challanging tasks and provide it to the agent. This is a form of active learning.

It's especially usefull because it may prove that some tasks are easier/harder than we thought and the agent itself should decide what it wants to learn next to minimise the time it takes to train.

## Imitation Learning

In short, again, from what I understand, it's a supervised phase before the actual training phase when the agent is provided with state/action pairs that were performed by an experienced (or not) oracle (human, another algorithm). The agent first tries to predict the actions based on states in a supervised setting (classification) and then the knowledge can be somehow transfered to the actual agent. 

## Environment Ideas

Here is a list of some ideas for the environments we can implement in our thesis along with their advantages/disadvatanges and ways they can be sequenced into a curriculum. I will also try to come up with how the environment can be represented by a state.


### Racing Agent

An agent that navigates a racing tracks as fast as possible. 

**Why Yes**

Fairly simple but also interesting and visually pleasing. A lot of ways it can be sequenced. 

**Why Not**

Can be somewhat difficult to implement (but not *that* difficult). A lot of people have done it already (well, not the curriculum part at least)

**Sequencing**

The agent can first learn to drive on a straight track, then navigate curves, sharper and sharper turns, more and more complex tracks and finally it can learn to do all of it faster and faster. The target task could be one of a few (or many) predefined tasks that are somehow more difficult, include a new element that was not seen in the curriculum (otherwise one can argue the target task was already included in the curriculum). This is an important aspect of curricula in general - they must not include the target task.

**State Representation**

The agents has a sort of *camera* attached to it. It shoots rays in several/many directions calculated the distance they travelled until they reached a wall and creates a vector of values. The agent can/should also have knowledge of its speed and direction.

### Self-Driving Agent

Similar to the Racing Agent but in a city setting instead of a racing track.

**Why Yes**

*Very* interesting (real life uses). A lot of ways it can be sequenced and experimented on. 

**Why Not**

Could be very difficult to implement and efficiently train on. The curricula could not meet our expectations or the agent may not converge to an optimal policy at all.

**Sequencing**

Similar as before but more focued on traffic laws rather than fast track navigation. The trainer can first be trained to obey simple laws like speed limits, stop signs, traffic lights and keeping in lane. Then it can be trained to navigate crossings, avoid other cars/pedestrains etc. 

**State Representation**

Similar as before but with more knowledge about its environement e.g. whether there is a stop sign/traffic light in front of it, other cars position. 

If the target task is to drive from point A to point B the agent should also know the route (we should not expect it to figure it out on its own although it could also be an interesting challange, instead we should assume some kind of navigation data is provided (Google Maps)). For example the agent can know whether to turn left/right go straight on the next crossing. 

### Hackmatch Agent

There is a game called EXAPUNKS which has a mini game called Hackmatch. It's a block stacking game, the target is to stack 4 or more same-coloured blocks together to destroy them. More blocks appear every second so the agent must do it quickly (before they fill up the available space). It's a bit like Tetris but not much.

**Why Yes**

It's fun!

**Why Not**

No practical uses

**Sequencing**

Less colours, premade chunks of blocks (up to 3), smaller state/actions space - smaller game area

**State Representation**

A matrix of R x C x D where R and C are respectively numbers of rows and columns of the game area (essentialy how many blocks can it fit) and D is the number of colours. The matrix is one-hot encoded along the D dimension (zero vector means no block in that spot). The agent should also know it's position and what kind of block it's currently holding (if any).

### Environment Template

**Why Yes**

**Why Not**

**Sequencing**

**State Representation**