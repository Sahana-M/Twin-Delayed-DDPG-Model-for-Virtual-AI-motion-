# Twin Delayed DDPG Model for Virtual AI motion

* ***Reference*** :- Addressing Function approximation Error in Actor-Critic methods, by Scott Fujimoto,
Herke van Hoof, David Meger (2018)  [Paper.](https://arxiv.org/pdf/1802.09477.pdf)

TD3 has been a successor of Deep Deterministic Policy Gradient. Until recent times, DDPG
was the most sought-after algorithm for continuous locomotory agents like Robotics or
autonomous driving. Question arises as to why this has been replaced on fame grounds with
the TD3 model?

This is because DDPG model like many RL algorithms are unstable and highly reliant on
finding the correct hyperparameters for the agent model. This is caused by the algorithm that
has been continuously over estimating the Q values of the critic network. These estimation
errors sum up leading to a faulty model leading to the model falling into local optima and
experience catastrophic forgetting. Our TD3 model focuses on reducing this overestimation
bias. This is solved in three ways,
* With 2 pair of critic deep neural networks
* With a pair of actor network for the delaying part
* Actor noise regularization part

## Architecture
![TD3](https://github.com/Sahana-M/Images/blob/master/TD3.png)

