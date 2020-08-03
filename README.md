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


#### Working of the Actor Critic Model
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


#### Polyak Averaging
Judges (Critic target 1 & Critic target 2) may be wrong sometimes. They get updated for every two iterations.
Model is tested in a real-time environment where Critic Model 1 and Critic Model 2 are engaged. 
So, in order to correct the targets, we will be copying 10% of the parameters to the Critic Target 1 and 
Critic Target 2 every two iterations

                θ(i)t ← r*θ(i) + (1-r)*θ(i)t


## Twin Delayed Deep Deterministic Gradient Policy model implementation

### Q-learning part
Training process - We run a full episode with the first 10,000 actions eyed randomly and with
the actions played by the actor model. 

The reason is that the TD3 model is off-policy and we are learning Q values and Deep Q
learning is also an off-policy model which leverages an experience replay memory from
which the model is going to extract me transitions on which the model is going to learn the Q
values and that’s the reason why we have to get a full episode in order to populate the
experience replay memory with already tons of transitions.

These transitions must be informed of random input to keep a balance between exploration &
exploitation and by adding noise to the transitions.


### Steps
* Step - 1: The Experience Replay memory is initiated with a size of 20,000 transitions.
* Step - 2: The Neural networks for both Actor model and Actor Target are built.
* Step - 3: The Neural networks for both Critic Model and Critic target are built (2 pairs).
* Step - 4: We sample a batch of transitions (s, s’, a, r) from the memory. Then for each element.
of the batch (100 transitions in a batch). We divide each column (Present state, Next state, Action, Reward) into separate batches.
* Step - 5: The Actor target plays the next action a’, from the next state s’.
* Step - 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment.
* Step - 7: The two Critic targets take each the couple (s’, a’) as input and return the two Qvalues Qt1(s’, a’) and Qt2(s’, a’) as outputs.
* Step - 8: We keep the minimum of these two Q values min (Qt1, Qt2). It represents the
approximated value of the next state. (Taking the minimum Q value prevents too optimistic
estimates of that value of state, this helps in stabilizing the optimization)
* Step - 9: We get the final target of the two Critic models which is

               Qt = r + γ * min ( Qt1, Qt2 ) 

* Step - 10: The two Critic models take the couple (s,a) as input and return two Q values
Q1(s,a) and Q2(s,a) as outputs.
* Step - 11: We compute the loss coming from the two Critic models:

              Critic loss = MSE_loss(Q1(s,a), Qt) + MSE_loss(Q2(s,a), Qt)
      
* Step - 12: We backpropagate this Critic loss and update the parameters of the two Critic
models with a SGD optimizer (here Adam optimizer)


### Policy learning part
* Step - 13: Once every two iterations, we update our Actor model by performing gradient
ascent on the output of the first Critic model, by differentiating the weights of the Actor model
by the Q value of the Critic model.
* Step - 14: Still once every two iterations we update the weights of the Actor target by Polyak
averaging

              θ(i)t ← r*θ(i) + (1-r)*θ(i)t

 ```
 where,
   Ѳit - weights of the Actor target
   Ѳi - weights of the Actor model
   ґ - very small value
```

* Step - 15: Still once every two iterations we update the weights of the Critic targets by Polyak
averaging

              Ѡ(i)t ← ґ*Ѡ(i) + (1-ґ)*Ѡ(i)t

```
where,
    Ѡit - weights of the Critic target
    Ѡi - weights of the Critic model
    ґ - very small value
```
