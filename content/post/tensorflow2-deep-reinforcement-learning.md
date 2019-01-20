+++
title = "Deep Reinforcement Learning with TensorFlow 2.0"

date = 2019-01-20
draft = false

authors = []

tags = []

summary = "In this tutorial I will showcase the upcoming TensorFlow 2.0 features through the lense of deep reinforcement learning (DRL) by implementing an advantage actor-critic (A2C) agent to solve the classic CartPole-v0 environment. While the goal is to showcase TensorFlow 2.0, I will do my best to make the DRL aspect approachable as well, including a brief overview of the field."

+++

In this tutorial I will showcase the upcoming TensorFlow 2.0 features through the lense of deep reinforcement learning (DRL) by implementing an advantage actor-critic (A2C) agent to solve the classic CartPole-v0 environment. While the goal is to showcase TensorFlow 2.0, I will do my best to make the DRL aspect approachable as well, including a brief overview of the field. 

In fact since the main focus of the 2.0 release is making developers’ lives easier, it’s a great time to get into DRL with TensorFlow - our full agent source is under 150 lines! Code is available as a notebook [here](https://github.com/inoryy/tensorflow2-deep-reinforcement-learning) and online on Google Colab [here](https://colab.research.google.com/drive/12QvW7VZSzoaF-Org-u-N6aiTdBN5ohNA).

## Setup

As TensorFlow 2.0 is still in experimental stage, I recommend installing it in a separate (virtual) environment. I prefer [Anaconda](https://www.anaconda.com/download), so I’ll illustrate with it:

```
> conda create -n tf2 python=3.6
> source activate tf2
> pip install tf-nightly-2.0-preview # tf-nightly-gpu-2.0-preview for GPU version
```

Let’s quickly verify that everything works as expected:

```python
>>> import tensorflow as tf
>>> print(tf.__version__)
1.13.0-dev20190117
>>> print(tf.executing_eagerly())
True
```

Don’t worry about the 1.13.x version, just means that it’s an early preview. What’s important to note here is that we’re in eager mode by default!

```python
>>> print(tf.reduce_sum([1, 2, 3, 4, 5]))
tf.Tensor(15, shape=(), dtype=int32)
```

If you’re not yet familiar with eager mode, then in essence it means that computation is executed at runtime, rather than through a pre-compiled graph. You can find a good overview in the [TensorFlow documentation](https://www.tensorflow.org/tutorials/eager/eager_basics).

## Deep Reinforcement Learning

Generally speaking, reinforcement learning is a high level framework for solving sequential decision making problems. A RL `agent` navigates an `environment` by taking `actions` based on some `observations`, receiving `rewards` as a result. Most RL algorithms work by maximizing sum of rewards an agent collects in a `trajectory`, e.g. during one in-game round.  

The output of an RL based algorithm is typically a `policy` - a function that maps states to actions. A valid policy can be as simple as a hard-coded no-op action. Stochastic policy is represented as a conditional probability distribution of actions, given some state.
[![](https://i.imgur.com/fUcDHVt.png)](http://web.stanford.edu/class/cs234/index.html)

#### Actor-Critic Methods

RL algorithms are often grouped based on the objective function they are optimized with. `Value-based` methods, such as [DQN](https://deepmind.com/research/dqn/), work by reducing the error of the expected state-action values. 

`Policy Gradients` methods directly optimize the policy itself by adjusting its parameters, typically via gradient descent. Calculating gradients fully is usually intractable, so instead they are often estimated via monte-carlo methods. 

The most popular approach is a hybrid of the two: `actor-critic` methods, where agents policy is optimized through policy gradients, while value based method is used as a bootstrap for the expected value estimates.

#### Deep Actor-Critic Methods

While much of the fundamental RL theory was developed on the tabular cases, modern RL is almost exclusively done with function approximators, such as artificial neural networks. Specifically, an RL algorithm is considered “deep” if the policy and value functions are approximated with `deep neural networks`.

[![](https://i.imgur.com/gsXfI91.jpg)](https://www.researchgate.net/publication/319121340_Enabling_Cognitive_Smart_Cities_Using_Big_Data_and_Machine_Learning_Approaches_and_Challenges)

#### (Asynchronous) Advantage Actor-Critic

Over the years, a number of improvements have been added to address sample efficiency and stability of the learning process. 

First, gradients are weighted with `returns`: discounted future rewards, which somewhat alleviates the credit assignment problem, and resolves theoretical issues with infinite timesteps. 

Second, an `advantage function` is used instead of raw returns. Advantage is formed as the difference between returns and some baseline (e.g. state-action estimate) and can be thought of as a measure of how good a given action is compared to some average. 

Third, an additional `entropy maximization` term is used in objective function to ensure agent sufficiently explores various policies. In essence, entropy measures how random a probability distribution is, maximized with uniform distribution.

Finally, `multiple workers` are used in `parallel` to speed up sample gathering while helping decorrelate them during training. 

Incorporating all of these changes with deep neural networks we arrive at the two of the most popular modern algorithms: (asynchronous) advantage actor critic, or `A3C/A2C` for short. The difference between the two is more technical than theoretical: as the name suggests, it boils down to how the parallel workers estimate their gradients and propagate them to the model.
[![](https://i.imgur.com/CL0w8rl.png)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

With this I will wrap up our tour of DRL methods as the focus of the blog post is more on the TensorFlow 2.0 features. Don’t worry if you’re still unsure about the subject, things should become clearer with code examples. If you want to learn more then one good resource to get started is [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest).

## Advantage Actor-Critic with TensorFlow 2.0

Now that we’re more or less on the same page, let’s see what it takes to implement the basis of many modern DRL algorithms: an actor-critic agent, described in previous section. For simplicity, we won’t implement parallel workers, though most of the code will have support for it. An interested reader could then use this as an exercise opportunity.

As a testbed we will use the [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) environment. Somewhat simplistic, it's still a great option to get started with. I always rely on it as a sanity check when implementing RL algorithms.

#### Policy & Value via Keras Model API

First, let's create the policy and value estimate NNs under a single model class:

```python
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
```

And let's verify the model works as expected:

```python
import gym

env = gym.make('CartPole-v0')
model = Model(num_actions=env.action_space.n)

obs = env.reset()
# no feed_dict or tf.Session() needed at all
action, value = model.action_value(obs[None, :])
print(action, value) # [1] [-0.00145713]
```

Things to note here:

- Model layers and execution path are defined separately
- There is no "input" layer, model will accept raw numpy arrays
- Two computation paths can be defined in one model via functional API
- A model can contain helper methods such as action sampling
- In eager mode everything works from raw numpy arrays

#### Random Agent

Now we can move on to the fun stuff - the `A2CAgent` class. First, let's add a `test` method that runs through a full episode and returns sum of rewards.

```python
class A2CAgent:
    def __init__(self, model):
        self.model = model

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward
```

Let's see how much our model scores with randomly initialized weights: 

```python
agent = A2CAgent(model)
rewards_sum = agent.test(env)
print("%d out of 200" % rewards_sum) # 18 out of 200
```

Not even close to optimal, time to get to the training part!

#### Loss / Objective Function

As I've described in the DRL overview section, an agent improves its policy through gradient descent based on some loss (objective) function. In actor-critic we train on three objectives: improving policy with advantage weighted gradients plus entropy maximization, and minizing value estimate errors.

```python
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms
        self.params = {'value': 0.5, 'entropy': 0.0001}
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    def test(self, env, render=True):
        # unchanged from previous section
        ...

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # polymorphic CE loss function that supports sparse and weighted options
        # from_logits argument ensures transformation into normalized probabilities
        cross_entropy = kls.CategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        # thus under the hood a sparse version of CE loss will be executed
        actions = tf.cast(actions, tf.int32)
        policy_loss = cross_entropy(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = cross_entropy(logits, logits)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss
```

And we're done with the objective functions! Note how compact the code is: there's almost more comment lines than code itself.

#### Agent Training Loop

Finally, there's the train loop itself. It's relatively long, but fairly straightforward: collect samples, calculate returns and advantages, and train the model on them.

```python
class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
        # unchanged from previous section
        ...
        
   def train(self, env, batch_sz=32, updates=1000):
        # storage helpers for a single batch of data
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + env.observation_space.shape)
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
        return ep_rews

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def test(self, env, render=True):
        # unchanged from previous section
        ...

    def _value_loss(self, returns, value):
        # unchanged from previous section
        ...

    def _logits_loss(self, acts_and_advs, logits):
        # unchanged from previous section
        ...
```

#### Training & Results

We're now all set to train our single-worker A2C agent on CartPole-v0! Training process shouldn't take longer than a couple of minutes. After training is complete you should see an agent successfully achieve the target 200 out of 200 score.

```python
rewards_history = agent.train(env)
print("Finished training, testing...")
print("%d out of 200" % agent.test(env)) # 200 out of 200
```

![](https://thumbs.gfycat.com/SoupyConsciousGrayling-size_restricted.gif)

In the source code I include some additional helpers that print out running episode rewards and losses, along with basic plotter for the rewards_history.

![](https://i.imgur.com/cFwQgPB.png)

## Static Computational Graph

With all of this eager mode excitement you might wonder if static graph execution is even possible anymore. Of course it is! Moreover, it takes just one additional line to enable it!

```python
with tf.Graph().as_default():
    print(tf.executing_eagerly()) # False

    model = Model(num_actions=env.action_space.n)
    agent = A2CAgent(model)

    rewards_history = agent.train(env)
    print("Finished training, testing...")
    print("%d out of 200" % agent.test(env)) # 200 out of 200
```

There's one caveat that during static graph execution we can't just have Tensors laying around, which is why we needed that trick with CategoricalDistribution during model definition. In fact, while I was looking for a way to execute in static mode, I discovered one interesting low level detail about models built through the Keras API...

## One More Thing…

Remember when I said TensorFlow runs in eager mode by default, even proving it with a code snippet? Well, I lied! Kind of. 

If you use Keras API to build and manage your models then it will attempt to compile them as static graphs under the hood. So what you end up getting is the performance of static computational graphs with flexibility of eager execution.

You can check status of your model via the `model.run_eagerly` flag. You can also force eager mode by setting this flag to `True`, though most of the times you probably don’t need to - if Keras detects that there's no way around eager mode, it will back off on its own.

To illustrate that it’s indeed running as a static graph here's a simple benchmark:


```python
# create a 100000 samples batch
env = gym.make('CartPole-v0')
obs = np.repeat(env.reset()[None, :], 100000, axis=0)
```

#### Eager Benchmark

```python
%%time

model = Model(env.action_space.n)
model.run_eagerly = True

print("Eager Execution:  ", tf.executing_eagerly())
print("Eager Keras Model:", model.run_eagerly)

_ = model(obs)

######## Results #######

Eager Execution:   True
Eager Keras Model: True
CPU times: user 639 ms, sys: 736 ms, total: 1.38 s
```

#### Static Benchmark

```python
%%time

with tf.Graph().as_default():
    model = Model(env.action_space.n)

    print("Eager Execution:  ", tf.executing_eagerly())
    print("Eager Keras Model:", model.run_eagerly)

    _ = model.predict(obs)

######## Results #######

Eager Execution:   False
Eager Keras Model: False
CPU times: user 793 ms, sys: 79.7 ms, total: 873 ms
```

#### Default Benchmark

```python
%%time

model = Model(env.action_space.n)

print("Eager Execution:  ", tf.executing_eagerly())
print("Eager Keras Model:", model.run_eagerly)

_ = model.predict(obs)

######## Results #######

Eager Execution:   True
Eager Keras Model: False
CPU times: user 994 ms, sys: 23.1 ms, total: 1.02 s
```

As you can see eager mode is behind static mode, and by default our model was indeed executing statically, more or less matching explicit static graph execution.

## Conclusion

Hopefully this has been an illustrative tour of both DRL and the things to come in TensorFlow 2.0. Note that this is still just a nightly preview build, not even a release candidate. Everything is subject to change and if there’s something about TensorFlow you especially dislike (or like :) ) , [let the developers know](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers)!

A lingering question people might have is if TensorFlow is better than PyTorch? Maybe. Maybe not. Both are great libraries, so it is hard to say one way or the other. If you’re familiar with PyTorch, you probably noticed that TensorFlow 2.0 not only caught up, but also avoided some of the PyTorch API pitfalls. 

In either case what is clear is that this competition has resulted in a net-positive outcome for both camps and I am excited to see what will become of the frameworks in the future.
