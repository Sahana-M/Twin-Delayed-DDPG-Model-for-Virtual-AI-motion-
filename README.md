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


This is a three-layered architecture with 6 neural networks which use ReLU
function as their activation function in the hidden layers and tanh function in their output layer.

* Actor Target 1 is placed in environment and it observes everything and takes actions and
receives reward based on the quality of action performed.
* Critic Target 1 judges the action by Actor target and gives the Q-value (Quality of action) to the actions
taken by the Actor Target. The Critic target neural network takes input the state and action in
return outputs the Q-value
* Critic Target 1 and Critic Target 2 are the Judges and the judgement is based on the Q-values
These Targets are not exposed to the Real-time Environment
* Critic Model 1 and Critic Model 2 will be learning based on the judgements of the Critic
Target 1 and Critic Target 2.
* Critic Model 1 and Critic Model 2 also predict the Q-value, their comparision happens with the
* Critic Target 1 and Critic Target 2, this is computed as loss and weights are updated using Backpropagation.
* Actor Model is the last one to learn. So, everything is based on the Q-value.

