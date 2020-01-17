+++
title = "Deep Reinforcement Learning With TensorFlow 2.1"

date = 2019-01-20
lastmod = 2020-01-13
draft = false

authors = []

tags = []

summary = """In this tutorial, I will give an overview of the TensorFlow 2.x features through the lens of deep reinforcement learning (DRL)
by implementing an advantage actor-critic (A2C) agent, solving the classic CartPole-v0 environment.
While the goal is to showcase TensorFlow 2.x, I will do my best to make DRL approachable as well, including a birds-eye overview of the field."""

+++

{{% toc %}}
<script type="text/javascript" src="/js/fix-toc.js"></script>

## Introduction

In this tutorial, I will give an overview of the TensorFlow 2.x features through the lens of deep reinforcement learning (DRL)
by implementing an advantage actor-critic (A2C) agent, solving the classic CartPole-v0 environment.
While the goal is to showcase TensorFlow 2.x, I will do my best to make DRL approachable as well,
including a birds-eye overview of the field.

In fact, since the main focus of the 2.x release is making life easier for the developers,
it’s a great time to get into DRL with TensorFlow.
For example, the source code for this blog post is under 150 lines, including comments!  
Code is available on GitHub [here](https://github.com/inoryy/tensorflow2-deep-reinforcement-learning)
and as a notebook on Google Colab [here](https://colab.research.google.com/drive/1XoHmGiwo2eUN-gzSVLRvE10fIf_ycO1j).

## Setup

To follow along, I recommend setting up a separate (virtual) environment.  
I prefer [Anaconda](https://www.anaconda.com/download), so I’ll illustrate with it:

```
> conda create -n tf2 python=3.7
> conda activate tf2
> pip install tensorflow=2.1 
```

Let us quickly verify everything works as expected:

```python
>>> import tensorflow as tf
>>> print(tf.__version__)
2.1.0
```

Note that we are now in eager mode by default!

```python
>>> print(tf.executing_eagerly())
True
>>> print("1 + 2 + 3 + 4 + 5 =", tf.reduce_sum([1, 2, 3, 4, 5]))
1 + 2 + 3 + 4 + 5 = tf.Tensor(15, shape=(), dtype=int32)
```

If you are not yet familiar with eager mode, then, in essence, it means that computation executes at runtime,
rather than through a pre-compiled graph.
You can find a good overview in the [TensorFlow documentation](https://www.tensorflow.org/tutorials/eager/eager_basics).

#### GPU Support

One great thing about specifically TensorFlow 2.1 is that there is no more hassle with separate CPU/GPU wheels!
TensorFlow now supports both by default and targets appropriate devices at runtime. 

The benefits of Anaconda are immediately apparent if you want to use a GPU.
Setup for all the necessary CUDA dependencies is just one line:

```
> conda install cudatoolkit=10.1
``` 

You can even install different CUDA toolkit versions in separate environments!


## Reinforcement Learning

Generally speaking, reinforcement learning is a high-level framework for solving sequential decision-making problems.
An RL `agent` navigates an `environment` by taking `actions` based on some `observations`, receiving `rewards` as a result.
Most RL algorithms work by maximizing the expected total rewards an agent collects in a `trajectory`, e.g., during one in-game round.

The output of an RL algorithm is a `policy` -- a function from states to actions.  
A valid policy can be as simple as a hard-coded no-op action,
but typically it represents a conditional probability distribution of actions given some state.

![](https://i.imgur.com/fUcDHVt.png)
<span class="source">Figure: A general diagram of the RL training loop.</br>
Image via [Stanford CS234 (2019)](http://web.stanford.edu/class/cs234/index.html).</span>

RL algorithms are often grouped based on their optimization `loss function`.

`Temporal-Difference` methods, such as `Q-Learning`, reduce the error between predicted and actual state(-action) `values`.

`Policy Gradients` directly optimize the policy by adjusting its parameters.
Calculating gradients themselves is usually infeasible; instead, they are often estimated via `monte-carlo` methods.

The most popular approach is a hybrid of the two: `actor-critic` methods, where policy gradients optimize agent's policy,
and the temporal-difference method is used as a bootstrap for the expected value estimates.

#### Deep Reinforcement Learning 

While much of the fundamental RL theory was developed on the tabular cases,
modern RL is almost exclusively done with function approximators, such as `artificial neural networks`.
Specifically, an RL algorithm is considered `deep` if the policy and value functions are approximated with neural networks.

![](https://i.imgur.com/gsXfI91.jpg)
<span class="source">Figure: DRL implies ANN is used in the agent's model.</br>
Image via [Mohammadi et al (2018)](https://arxiv.org/abs/1810.04107).</span>

#### (Asynchronous) Advantage Actor-Critic

Over the years, several improvements were added to address sample efficiency and stability of the learning process.

First, gradients are weighted with `returns`: a discounted sum of future rewards,
which resolves theoretical issues with infinite timesteps,
and mitigates the `credit assignment problem` -- allocate rewards to the correct actions. 

Second, an `advantage function` is used instead of raw returns.
Advantage is formed as the difference between the returns and some `baseline`, which is often the value estimate,
and can be thought of as a measure of how good a given action is compared to some average. 

Third, an additional `entropy maximization` term is used in the objective function to ensure the agent
sufficiently explores various policies. In essence, entropy measures how *random* a given probability distribution is.
For example, entropy is highest in the uniform distribution.

Finally, multiple workers are used in `parallel` to speed up sample gathering while helping decorrelate them during training,
diversifying the experiences an agent trains on in a given batch.

Incorporating all of these changes with deep neural networks, we arrive at the two of the most popular modern algorithms:
(asynchronous) advantage actor critic, or `A3C/A2C` for short. The difference between the two is more technical than theoretical.
As the name suggests, it boils down to how the parallel workers estimate their gradients and propagate them to the model.

![](https://i.imgur.com/CL0w8rl.png)
<span class="source">Image via [Juliani A. (2016)](http://bit.ly/2uAJm2S).</span>

With this, we wrap up our tour of the DRL methods and move on to the focus of the blog post is more on the TensorFlow 2.x features.
Don’t worry if you’re still unsure about the subject; things should become clearer with code examples.  
If you want to learn more, one excellent resource is [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest).

## Advantage Actor-Critic With TensorFlow 2.1

Now that we are more or less on the same page, let’s see what it takes to implement the basis of many modern DRL algorithms:
an actor-critic agent, described in the previous section. Without parallel workers (for simplicity), though
most of the code would be the same.

As a testbed, we are going to use the [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) environment.
Somewhat simplistic, it is still a great option to get started. In fact, I often rely on it as a sanity check when implementing RL algorithms.

#### Policy & Value Models via Keras API 

First, we create the policy and value estimate NNs under a single model class:

```python
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl


class ProbabilityDistribution(tf.keras.Model):
  def call(self, logits, **kwargs):
    # Sample a random categorical action from the given logits.
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    # Note: no tf.get_variable(), just simple Keras API!
    self.hidden1 = kl.Dense(128, activation='relu')
    self.hidden2 = kl.Dense(128, activation='relu')
    self.value = kl.Dense(1, name='value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name='policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    # Inputs is a numpy array, convert to a tensor.
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    # Executes `call()` under the hood.
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    # Another way to sample actions:
    #   action = tf.random.categorical(logits, 1)
    # Will become clearer later why we don't use it.
    return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
```

And verify the model works as expected:

```python
import gym

env = gym.make('CartPole-v0')
model = Model(num_actions=env.action_space.n)

obs = env.reset()
# No feed_dict or tf.Session() needed at all!
action, value = model.action_value(obs[None, :])
print(action, value) # [1] [-0.00145713]
```

Things to note here:

- Model layers and execution path are defined separately
- There is no “input” layer; model accepts raw numpy arrays
- Two computation paths can exist in one model via functional API
- A model can contain helper methods such as action sampling
- In eager mode, everything works from raw numpy arrays

#### Agent Interface

Now we can move on to the fun stuff -- the agent class.
First, we add a `test` method that runs through a full episode,
keeping track of the rewards.

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

Now we can check how much the agent scores with randomly initialized weights: 

```python
agent = A2CAgent(model)
rewards_sum = agent.test(env)
print("%d out of 200" % rewards_sum) # 18 out of 200
```

Not even close to optimal, time to get to the training part!

#### Loss / Objective Function

As I have described in the RL section, an agent improves its policy through gradient descent based on some loss (objective) function.
In the A2C algorithm, we train on three objectives: improve policy with advantage weighted gradients, maximize the entropy, and minimize value estimate errors.

```python
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko


class A2CAgent:
  def __init__(self, model, lr=7e-3, value_c=0.5, entropy_c=1e-4):
    # Coefficients are used for the loss terms.
    self.value_c = value_c
    self.entropy_c = entropy_c

    self.model = model
    self.model.compile(
      optimizer=ko.RMSprop(lr=lr),
      # Define separate losses for policy logits and value estimate.
      loss=[self._logits_loss, self._value_loss])

  def test(self, env, render=False):
    # Unchanged from the previous section.
    ...

  def _value_loss(self, returns, value):
    # Value loss is typically MSE between value estimates and returns.
    return self.value_c * kls.mean_squared_error(returns, value)

  def _logits_loss(self, actions_and_advantages, logits):
    # A trick to input actions and advantages through the same API.
    actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

    # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
    # `from_logits` argument ensures transformation into normalized probabilities.
    weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

    # Policy loss is defined by policy gradients, weighted by advantages.
    # Note: we only calculate the loss on the actions we've actually taken.
    actions = tf.cast(actions, tf.int32)
    policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

    # Entropy loss can be calculated as cross-entropy over itself.
    probs = tf.nn.softmax(logits)
    entropy_loss = kls.categorical_crossentropy(probs, probs)

    # We want to minimize policy and maximize entropy losses.
    # Here signs are flipped because the optimizer minimizes.
    return policy_loss - self.entropy_c * entropy_loss
```

And we are done with the objective functions!
Note how compact the code is: there are almost more comment lines than code itself.

#### The Training Loop

Finally, there is the train loop itself. It is relatively long, but fairly straightforward:
collect samples, calculate returns and advantages, and train the model on them.

```python
class A2CAgent:
  def __init__(self, model, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
    # `gamma` is the discount factor
    self.gamma = gamma
    # Unchanged from the previous section.
    ...
  
  def train(self, env, batch_sz=64, updates=250):
    # Storage helpers for a single batch of data.
    actions = np.empty((batch_sz,), dtype=np.int32)
    rewards, dones, values = np.empty((3, batch_sz))
    observations = np.empty((batch_sz,) + env.observation_space.shape)

    # Training loop: collect samples, send to optimizer, repeat updates times.
    ep_rewards = [0.0]
    next_obs = env.reset()
    for update in range(updates):
      for step in range(batch_sz):
        observations[step] = next_obs.copy()
        actions[step], values[step] = self.model.action_value(next_obs[None, :])
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])

        ep_rewards[-1] += rewards[step]
        if dones[step]:
          ep_rewards.append(0.0)
          next_obs = env.reset()
          logging.info("Episode: %03d, Reward: %03d" % (
            len(ep_rewards) - 1, ep_rewards[-2]))

      _, next_value = self.model.action_value(next_obs[None, :])

      returns, advs = self._returns_advantages(rewards, dones, values, next_value)
      # A trick to input actions and advantages through same API.
      acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

      # Performs a full training step on the collected batch.
      # Note: no need to mess around with gradients, Keras API handles it.
      losses = self.model.train_on_batch(observations, [acts_and_advs, returns])

      logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

    return ep_rewards

  def _returns_advantages(self, rewards, dones, values, next_value):
    # `next_value` is the bootstrap value estimate of the future state (critic).
    returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

    # Returns are calculated as discounted sum of future rewards.
    for t in reversed(range(rewards.shape[0])):
      returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
    returns = returns[:-1]

    # Advantages are equal to returns - baseline (value estimates in our case).
    advantages = returns - values

    return returns, advantages

  def test(self, env, render=False):
    # Unchanged from the previous section.
    ...

  def _value_loss(self, returns, value):
    # Unchanged from the previous section.
    ...

  def _logits_loss(self, actions_and_advantages, logits):
    # Unchanged from the previous section.
    ...

```

#### Results

We are now all set to train our single-worker A2C agent on CartPole-v0!
The training process should take a couple of minutes.
After the training is complete, you should see an agent achieve the target 200 out of 200 score.

```python
rewards_history = agent.train(env)
print("Finished training, testing...")
print("%d out of 200" % agent.test(env)) # 200 out of 200
```

![](https://thumbs.gfycat.com/SoupyConsciousGrayling-size_restricted.gif)

In the source code, I include some additional helpers that print out running episode rewards and losses,
along with basic plotter for the rewards history.

![](https://i.imgur.com/cFwQgPB.png)

## Static Computational Graph

With all of this eager mode excitement, you might wonder if using a static graph is even possible anymore.
Of course, it is! And it takes just one line!

```python
with tf.Graph().as_default():
  print(tf.executing_eagerly()) # False

  model = Model(num_actions=env.action_space.n)
  agent = A2CAgent(model)

  rewards_history = agent.train(env)
  print("Finished training, testing...")
  print("%d out of 200" % agent.test(env)) # 200 out of 200
```

There is one caveat though: during static graph execution, we can not just have Tensors laying around,
which is why we needed that trick with the separate `ProbabilityDistribution` model definition.
In fact, while I was looking for a way to execute in static mode,
I discovered one interesting low-level detail about models built through the Keras API...

## One More Thing…

Remember when I said TensorFlow runs in eager mode by default, even proving it with a code snippet? Well, I lied! Kind of.

If you use Keras API to build and manage your models, then it attempts to compile them as static graphs under the hood.
So what you end up with is the performance of static graphs with the flexibility of eager execution.

You can check the status of your model via the `model.run_eagerly` flag.
You can also force eager mode by manually setting it, though most of the times you probably don’t need to --
if Keras detects that there is no way around eager mode, it backs off on its own.

To illustrate that it is running as a static graph here is a simple benchmark:

```python
# Generate 100k observations to run benchmarks on.
env = gym.make('CartPole-v0')
obs = np.repeat(env.reset()[None, :], 100000, axis=0)
```

**Eager Benchmark**

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

**Static Benchmark**

```python
%%time

with tf.Graph().as_default():
    model = Model(env.action_space.n)

    print("Eager Execution:  ", tf.executing_eagerly())
    print("Eager Keras Model:", model.run_eagerly)

    _ = model.predict_on_batch(obs)

######## Results #######

Eager Execution:   False
Eager Keras Model: False
CPU times: user 793 ms, sys: 79.7 ms, total: 873 ms
```

**Default Benchmark**

```python
%%time

model = Model(env.action_space.n)

print("Eager Execution:  ", tf.executing_eagerly())
print("Eager Keras Model:", model.run_eagerly)

_ = model.predict_on_batch(obs)

######## Results #######

Eager Execution:   True
Eager Keras Model: False
CPU times: user 994 ms, sys: 23.1 ms, total: 1.02 s
```

As you can see, eager mode is behind static, and by default, our model was indeed executed statically,
almost matching the explicitly static execution.

## Conclusion

Hopefully, this has been an illustrative tour of both DRL and the shiny new things in TensorFlow 2.x.
Note that many of the design choice discussions are [open to the public](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers),
and everything is subject to change. If there is something about TensorFlow, you especially dislike (or like :) ), let the developers know!

A lingering question people might have is if TensorFlow is better than PyTorch? Maybe. Maybe not.
Both are excellent libraries, so it is hard to say one way or the other. If you are familiar with PyTorch,
you probably noticed that TensorFlow 2.x has caught up and arguably avoided some of the PyTorch API pitfalls. 

At the same time, I think it would be fair to say that PyTorch was affected by the design choices of TensorFlow.
What is clear is that this "competition" has resulted in a net-positive outcome for both camps!

