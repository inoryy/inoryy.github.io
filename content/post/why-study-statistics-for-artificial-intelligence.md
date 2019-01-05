+++
title = "Why I Majored in Statistics for a Career in Artificial Intelligence"

date = 2019-01-05
#lastmod = 2018-01-13T00:00:00
draft = false

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = []

tags = []

summary = "Undergraduate major is often the first significant career decision a person makes in his life. As artificial intelligence (AI) becomes more and more ingrained in our society, many people begin to consider a career in AI as a viable choice in their life. However, it is still very rare to have an undergraduate degree fully dedicated to AI, so people opt for what they perceive to be the next best thing - computer science. But I believe there is a better alternative: statistics, and in this blog post I will try to explain why, based on my own example."

#[header]
#image = "headers/getting-started.png"
#caption = "Image credit: [**Academic**](https://github.com/gcushen/hugo-academic/)"
+++

Undergraduate major is often the first significant career decision a person makes in his life. As artificial intelligence (AI) becomes more and more ingrained in our society, many people begin to consider a career in AI as a viable choice in their life. However, it is still very rare to have an undergraduate degree fully dedicated to AI, so people opt for what they perceive to be the next best thing - computer science. But I believe there is a better alternative: statistics, and in this blog post I will try to explain why, based on my own example.

In the recent years AI and its many subfields like machine learning (ML) have exploded in popularity and are on track to pretty much take over every industry out there. And people have noticed: the introductory course Machine Learning (CS229) at Stanford had over a thousand students enrolled in the fall of 2017!

{{< tweet 912382154155352064 >}}

Well, I want to let you in on a little secret: majority of AI/ML, including the hip new trend you’ve probably heard about called deep learning, is just applied statistics in disguise: many ML techniques and algorithms are either fully borrowed from or heavily rely on the theory from statistics.

[![](https://imgs.xkcd.com/comics/machine_learning.png)](https://xkcd.com/1838/)

Unfortunately, statistics is lacking in their PR department, which causes many people to misunderstand the field. When I said I am majoring in statistics, many of my peers reaction was confusion at best, with some even assuming I was joking. In fact I was often looked down upon by both maths and CS majors: mathematicians considered stats to be not “pure” enough, whereas CS people thought it’s not engineering-oriented enough. What’s funny is that I actually agree with both of those camps, but I believe those to be pros rather than cons. So let’s review some of the core subjects I’ve taken during my undergraduate studies and how they have helped me with AI/ML.

**Mathematical Analysis.** You’ve probably heard of or even taken the “practical” alternative to it - Calculus, which is an okay subject, but in my opinion by not focusing on the theory behind the various theorems and lemmas a student never actually builds an intuitive understanding. And boy does it help in AI/ML. The topics at the heart of MA - continuity and differentiability, are also what is behind most of AI/ML algorithms.

**Probability Theory and Statistics.** Again, some version of this subject is likely taught in CS degrees as well, but theory is typically avoided. I’ve had no trouble diving head first into reinforcement learning thanks to deep and intuitive understanding of random variables and their estimates, expectations, distributions, and so on.

**Numerical Methods.** Speaking of what is behind most of AI/ML, this subject tackles the questions of function optimization and approximation. And if function approximation sounds alien to you, then perhaps you’ve heard of its special case - artificial neural networks.

**Matrix Calculus.** And while we’re on the subject of artificial neural networks, you’ve probably heard that they are represented as a chain of differentiable matrix operations. Well, here is a whole subject dedicated to understanding how to transfer your multivariate differentiation theory into the world of linear algebra.

**Monte Carlo Methods.** Have you ever wondered how probability theory is applied in practice? How can your computer generate random variables from any distribution? Well, this subject covers this and much more. And if you are into reinforcement learning then this course is probably the most important one to take as it covers a large chunk of the theory behind it. For example, the REINFORCE family of algorithms are built on the monte carlo methods.

**Stochastic Processes.** Speaking of theory behind RL, here's a whole subject dedicated to dealing with probability distributions over time. Markov Chains, Renewals, Queues, Brownian Motions, Gaussian Processes, ...

**Data Analysis.** Did I mention that majority of machine learning is actually applied statistics? This course intimately covers the theory behind what people would refer to as classical ML - from simple linear regression to generalized models.

**Experimental Design.** As we are starting to reach the limitations of our hardware the various ML/RL experiments become increasingly expensive in terms of wall-clock time. More and more people are now looking for ways to extract similar quality of information with less effort. Well, statisticians have been working on this problem for decades and you can learn all about it in this course.

But what about programming, you might ask? Well, with a balanced curriculum you actually get quite a fair share of computer science. In fact, with a couple of good elective choices you can cover most of the fundamental knowledge necessary to work as a software engineer if you ever wanted to switch. For example here are the CS courses I have had in my undergrad:

- Intro to Programming (Python)
- Object Oriented Programming (Java)
- Algorithms and Data Structures (Java)
- Database Systems (SQL)
- Operating Systems (Python / Java)
- Programming Languages (Prolog, Haskell, Scala, OCaml, C)

So where’s the catch? Well, the problem with having a subject that is so widely misunderstood and unpopular is that the recruiters looking at your resume might assume you’re one of those hippies that prefers pen & paper to a keyboard and pass you over for a "safe" computer science guy. Unfortunately to get around this I think it’s inevitable that you still have to get the desired stamp, either via double major or with a computer science focused masters degree.

In conclusion, I believe statistics to be the perfect major for a career in AI. As I am wrapping up my first semester of computer science masters I feel that I am often quite ahead of my peers specifically because of my undergraduate background and hopefully I have persuaded some of you to give it a shot!
