+++
title = "DeepMind, StarCraft II, and The Next Big Thing in AI"

date = 2019-01-24
draft = false

authors = []

tags = []

summary = "After winning against Go world champion Lee Sedol in 2016, DeepMind has announced the  next challenge they will focus on: StarCraft II. Today they are finally ready to unveil *something* at 6PM GMT and with this blog post I want to help both StarCraft players and AI researchers appreciate the scope of what they're about to experience. I will give a brief overview of the challenge, address some of the common misconceptions, and speculate a bit on what we'll see."

+++

Couple of days ago out of the blue DeepMind announced a StarCraft II related event, with many of the employees being quite excited about it on twitter. In just a few hours we will see what DeepMind has in store for us, but in the meantime let's take a step back and review how we got here and why it is so important.

**Update**: the (amazing) event has finished and DeepMind have released a [very detailed write-up](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/). See below for my thoughts on the event and the write-up.

# DeepMind

At the end of 2013, DeepMind was a small and relatively unknown startup, founded three years prior with one simple goal: "to solve intelligence". Although not yet famous with the general public, they have made a lot of noise in the world of deep reinforcement learning by introducing [DQN](https://deepmind.com/research/dqn/) - an algorithm, capable of matching or even surpassing humans at playing the Atari video games, while observing the game from raw pixels similar to how a human would. In the early 2014 DeepMind was bought by Google Inc. for $500 million, many speculating that deal was ensured by this one achievement.

<figure>
  <img src="https://raw.githubusercontent.com/kuz/DeepMind-Atari-Deep-Q-Learner/master/gifs/breakout.gif"  />
  <figcaption style="text-align: center;">
      <small>Atari game of Breakout, played by DQN agent. <a href="https://github.com/kuz/DeepMind-Atari-Deep-Q-Learner">Source</a>.</small>
  </figcaption>
</figure>

In March of 2016 DeepMind did it again, with their [AlphaGo](https://deepmind.com/research/alphago/) program decisively winning against Go world champion Lee Sedol, who held the title for a great number of years. While in hindsight this seems as a sure thing, three years ago this win came almost out of nowhere, with many researchers at the time were predicting that AI will not match human experts for another 10 years at least.

<figure>
  <img src="https://i.imgur.com/NNIEdt5.jpg"  />
  <figcaption style="text-align: center;">
      <small>DeepMind's AlphaGo winning vs Lee Sedol. <a href="https://www.engadget.com/2016/03/12/watch-alphago-vs-lee-sedol-round-3-live-right-now/">Source</a>.</small>
  </figcaption>
</figure>

# StarCraft II

Shortly after DeepMind's AlphaGo victory, DeepMind were already prepared to announce the next challenge they will be tackling: the StarCraft II video game. StarCraft II is a real-time strategy game that requires making split second decisions on strategical, tactical, and economical levels, with short-term and long-term goals, throughout the whole duration of the game. Together with its predecessor StarCraft: Brood War, it has a rich competitive e-sports history spanning over 20 years, and is actively played by millions to this day.

<figure>
  <img src="https://i.imgur.com/yaOixgL.png"  />
  <figcaption style="text-align: center;">
      <small>Serral securing his world championship title at WCS 2018. <a href="https://youtu.be/ZO9kqMGK190">Source</a>.</small>
  </figcaption>
</figure>

With much bigger state and action spaces, real-time component, partial observability, and longer game duration, this is a much more difficult task than any attempted before. In fact many AI researchers have actually attempted to tackle it, notably the StarCraft: Brood War bot [OverMind](https://arstechnica.com/gaming/2011/01/skynet-meets-the-swarm-how-the-berkeley-overmind-won-the-2010-starcraft-ai-competition/) was a collaborative effort of many talented scientists. Yet, when pitted against human players, the AI champion would be easily defeated even by relative novices to the game.

This massive challenge alone wasn't enough for DeepMind researchers, they decided to go one step further. They wanted to not only challenge human experts, but do so "on their terms". The AI will be observing the game in a similar way to humans, through image-like features, and will be acting by emulating keyboard and mouse commands as close as possible.

<figure>
  <img src="https://i.imgur.com/9FQkrkJ.jpg"  />
  <figcaption style="text-align: center;">
      <small>On the right: StarCraft II game as seen by the AI. <a href="https://deepmind.com/blog/deepmind-and-blizzard-open-starcraft-ii-ai-research-environment/">Source</a>.</small>
  </figcaption>
</figure>

Furthermore, the AI is limited in many aspects to match human experiences, e.g. it is allowed to take a limited number of actions per minute (APM) and can only access the same information a human would see on the screen or minimap. This means that the AI must learn to use in-game camera in order to effectively play.

<figure>
  <img src="https://storage.googleapis.com/deepmind-live-cms/documents/Oriol-Fig-Anim-170809-Optimised-r03.gif"  />
  <figcaption style="text-align: center;">
      <small>StarCraft II AI emulates actions as close to humans as possible. <a href="https://deepmind.com/blog/deepmind-and-blizzard-open-starcraft-ii-ai-research-environment/">Source</a>.</small>
  </figcaption>
</figure>

# The Next Big Thing in AI

A StarCraft II player might wonder what is special about this if there were relatively good AIs built into the game from release. The AI you encounter in-game acts based on some pre-defined set of rules, which means it can only play as well as the person who programmed it, reacting only to foreseen events. This makes the AI inflexible, and easy to exploit, among other things. 

In contrast, the AI DeepMind is working on learns the game by itself, essentially from scratch. At each step all it has is the game state it observed, the actions it can take, and some reward stimulus it received for a previous action. In general the approach is called reinforcement learning and [here](http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/) I give a brief overview of the field.

To get a feel for what it's like for an RL agent to "learn" the game, here is a video of an agent attempting to tackle a set of minigames released by DeepMind. On the left you can see an agent that just started the process and is exploring all the various actions it can take, and on the right is the same agent after a fair bit of such experiments (about 50 million steps on average).

{{< youtube gEyBzcPU5-w >}}
<br/>

An outside reader might wonder why AI researchers focus so much on games to begin with. The answer is simple: real-world is just too complex for an AI system at this point - one must first learn to walk before he gets to fly. Specifically, games have easily defined rules that can be described to an AI system and are typically easy to run in parallel on a computer, speeding up the learning process.

This however does not mean that current work will be useless in the future. For example, while "playing" with StarCraft II, DeepMind researchers produced a number of general-purpose results such as the [population based training](https://deepmind.com/blog/population-based-training-neural-networks/), and [relational deep reinforcement learning](https://openreview.net/forum?id=HkxaFoC9KQ). So while developing an AI capable of winning in StarCraft II with human-like restrictions *probably* won't give us [AGI](https://en.wikipedia.org/wiki/Artificial_general_intelligence), it will definitely bring us an inch closer.


# Predictions and Speculations

In this section I will attempt to predict some of the things we will see at the event, along with the nitty-gritty details "under the hood" of the AI, which might get a bit technical. Note that I have no insider information, this is simply my guesses.

For the event itself I believe it will be a Protoss vs Protoss showmatch between DeepMind's AI and some high level player, either ex-pro or possibly even current pro, however not best of the best tier. I would be *really* surprised if they are already at a level to challenge world champions.

The core algorithm they will employ will be an `actor-critic` variant called [IMPALA](https://deepmind.com/blog/impala-scalable-distributed-deeprl-dmlab-30/), fused with `attention` mechanism as described in [Relational DRL](https://openreview.net/forum?id=HkxaFoC9KQ). Network architecture will have `residual + convolutional` state encoder component and a number of `Conv LSTM` blocks generating the policy. It will also have a fair bit of `imitation learning` based pre-training prior to the DRL loop.

The `IMPALA` algorithm is essentially a distributed, asyncronous version of `A2C` I've described in [this](http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/) blog with a number of fixes (e.g. importance weighting) to address its "off-policyness" - the fact that samples are gathered with a different policy than the one being currently trained. If you're familiar with [openAI Five](https://blog.openai.com/openai-five/) then the core idea is similar to their asyncronous PPO: a (massively) distributed advantage actor-critic variant with tricks to correct for the distributional drift.

The `attention` mechanism has revolutionized the world of NLP, especially after the "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" article came out. In short, if we're dealing with sequential input / output then attention mechanism acts as importance weight for items in the given sequence when generating the next output. In NLP this is most often used for machine translation to give attention to words at different parts of the sentence. In StarCraft II this could be used to ensure the agent can quickly switch between tactical and strategical decision making. DeepMind has recently applied it to DRL in their [Relational DRL](https://openreview.net/forum?id=HkxaFoC9KQ) article. 

The `residual + convolutional` architecture makes sense to use given our state space, whereas `Conv LSTM` cells (NB! different from *CNN* LSTM) are starting to gain traction in DRL world, especially when observations are rich with spatial information.

During Blizzcon 2018 presentation Oriol Vinyals mentioned that they've used `imitation learning` for their agent - and it of course make sense, given that Blizzard is providing free access (to everybody!) to their massive dataset of replays. Of course it would be impossible to succesfully train a full agent from replays alone, but I think it's reasonable to use it for NN weights pre-training.

I also wouldn't be surprised if instead of fully end-to-end approach DeepMind will rely on something modular, e.g. as described [here](https://arxiv.org/abs/1811.03555), where individual modules are responsible for specific subsets of the game. Specifically, perhaps DeepMind relies on pre-defined rulesets for scouting and makes use of a simplified game engine for battle simulations that are used in MCTS solver to determine whether they should attack or fall back.

# Conclusion

Hopefully you're now as excited as I am about the upcoming event. Whether the AI is fully end to end or not and whether they will be able to win vs human experts or not, it is still a massive endeavor and should provide for a good show. And if you'd like to get into developing StarCraft II based AIs yourself, join our [SC2AI community on discord](https://discordapp.com/invite/Emm5Ztz)!

# Post-Event Write-Up

The stream was very exciting to watch, both as a player and a researcher. It was funny to see AlphaStar opt for wild strategies and then come out on top. It was also interesting to see it make mistakes such as killing its own units, very AI and human-like behavior at the same time.

To me the most impressive was the micro, and not just due to its ability to make split second decisions. The way AlphaStar knew how to pull back damaged stalkers to regenerate shields, the way it pulled its workers when it saw Oracles - really mind boggling that a single end-to-end neural network is capable of such a rich variety of tactical and somewhat long-term decision-making.

However, AlphaStar is still quite far from conquering StarCraft II universe. First, while Mana is no doubt a great player, he is not quite world champion caliber. Second, this is still a single matchup, whereas any human player would be expected to play vs all three races on the same level. Third, I'm not sure how I feel about having players go against a pool of AlphaStar(s) - I think it definitely makes sense to use for training, but during inference I'd prefer to see a single version used throughout the matches. Overall I would say AlphaStar right now is closer to the AlhaGo version that played vs Fan Hui than the one that won vs Lee Sedol.

Seems that I've correctly predicted the match-up and level of play, along with some of the approaches. Specifically, AlphaStar does indeed rely on `imitation learning`, `IMPALA`, and `attention` mechanism, though not quite as described in Relational DRL article. They also indeed use `LSTM`, but I am not so sure with regards to `convolutional` layers - there seems to be a bit of confusion as to what interface they ended up using. I've also briefly mentioned `population based training` - seems that DeepMind uses an advanced variant of it, hopefully we will see an article about it soon.

Of the things I've missed is the [transformer](https://arxiv.org/abs/1706.03762) body, which is a state-of-the-art architecture in machine translation. Very surprised to see it applied in DRL. They also use a relatively novel baseline for the `advantage function` in the PG loss, which they pulled from the [Counterfactual Multi-Agent Policy Gradients](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/foersteraaai18.pdf) article. This is noteworthy because for the longest time the value estimate of next state for baseline was the go-to approach of pretty much everybody in DRL.

Finally, they apply [pointer network](https://arxiv.org/abs/1506.03134) to the policy output, most likely as an efficient way to deal with variable length of action arguments. To me the use of `pointer networks` was quite surprising and somewhat ironic - I have actually [written an essay](http://inoryy.com/files/pointer_networks_essay.pdf) on this article and while it was an interesting subject, it never crossed my mind it could be applied in such a way to DRL policies. Although in retrospect I guess it makes sense.

During training they also rely on [self-imitation](http://proceedings.mlr.press/v80/oh18b/oh18b.pdf) and `experience replay`, which is quite interesting - seems they have finally perfected the combination of `actor-critic methods`, which are traditionally seen as `on-policy`, with the benefits of `off-policy` algorithms. Finally, they use [policy distillation](https://arxiv.org/pdf/1511.06295.pdf) which is probably how they were able to fit the final agents into a single machine for inference.

If by now your head is spinning from all the terminology, don't worry - mine is too. The takeaway message is that it took an impressive amount of very advanced approaches to achieve the level of play we have seen today and I am curious to see what happens next.
