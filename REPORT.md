## 1. Twin Delayed DDPG

Twin Delayed Deep Deterministic policy gradient algorithm (TD3) for Project 2

## 2. Background 

While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and 
other kinds of tuning. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, 
which then leads to the policy breaking, because it exploits the errors in the Q-function. Twin Delayed DDPG (TD3) is an algorithm 
that addresses this issue by introducing three critical tricks

Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.

Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.

Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for the policy to exploit Q-function errors by smoothing out Q along changes in action.

Together, these three tricks result in substantially improved performance over baseline DDPG.

## 3. Quick Facts
               
TD3 is an off-policy algorithm.
TD3 can only be used for environments with continuous action spaces.
The Spinning Up implementation of TD3 does not support parallelization.


## 3. Pseudo Code
![22222222](https://user-images.githubusercontent.com/75971822/135032186-9705ec2f-fd3a-4099-b370-c4206d7b3d62.jpg)

## 4. Networks
(1) Actor : state -> BatchNorm -> Linear(state_size, 256) -> BatchNorm -> LeakyRelu -> Linear(256, 128) -> BatchNorm -> LeakyRelu -> Linear(128, action_size) -> tanh

(2) Critic : state -> BatchNorm -> Linear(state_size, 256) -> Relu -> (concat with action) -> Linear(256+action_size, 128) -> Relu -> Linear(128, 1)

## 5. Parameters

n_episodes	2000
max_t	3000

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NUM_LEARN = 15          # number of learning 
NUM_TIME_STEP = 25      # every NUM_TIME_STEP do update
EPSILON = 1.0           # epsilon to noise of action
EPSILON_DECAY = 2e-6    # epsilon decay to noise epsilon of action
POLICY_DELAY = 3        # delay for policy update (TD3)

## 6. Result


![7979](https://user-images.githubusercontent.com/75971822/135193641-0ca1c18a-eb74-4286-a6b9-246fb8162895.png)