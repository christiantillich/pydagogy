#%% [markdown]
"""
# Objective

This is actually a lecture and a half. 16 spent half the class on ICA, and 
half the class on RL. And they don't really build into each other. So I'm moving
all that content here, and then we'll continue. Thus, this document will be entirely
RL. 

### RL Intro 

RL requires two things - state of the environment and an actor. The actor can
make certain decisions, and the state of the environment can tell if we're momentarily
successful or not. 

Supervised learning might make sense if we had access to some sort of
moment-by-moment data about what the _true best action_ was when trying to
fly the helicopter. But we don't, really. And RL says we don't need to - we can
design a system that rewards the machine pilot when it stays "on track", and
penalizes it when it does not, with excessive penalties for extremes like
crashing.

We might start with a reward function like
"""
#%% [python]
import latexify
@latexify.function(use_math_symbols = True)
def R(outcome):
    if outcome == 'win':
        return 1
    elif outcome == 'lose':
        return -1
    else:
        return 0
R

#%% [markdown]
"""
This is usually a pretty good start, but one of the first problems you're going
to run into is called the 'Credit Assignment Problem'. You could imagine a chess
algorithm making a very huge mistake early on, but the game still taking 40+ moves
to actually finish. You want the algo to identify the bad move early in the game, 
but your win/loss info comes 40 moves later. Which move does the computer assign
the loss to?

So keep this in mind, good RL algorithms need to be able to at least implicitly
solve this. 

### MDPs

MDPs are a way of framing the problem. An MDP is a 5-tuple output vector that
describes the whole system at any point. 

> `(S, A, P[sa], gamma, R)`

Here, 
* `S` is a set of states
* `A` is a set of actions
* `P[sa]` is the state transition probabilities, which say "if you take a certain
    action given state `s`, what's the probability of ending in a new state `s'`
* `gamma` is a discount facor
* `R` is a reward function

Chess and helicoptor piloting are really complex actions that have very complicated
MDPs, so instead we switch over to a robot navigating a grid and moving around
some obstacle, or through a maze. We end up with a 4x3 grid as an example, and
this corresponds to 11 different states the robot could be in, with the position
(2,2) being the obstacle itself. 

The action space itself is `(N,S,E,W)`, corresponding to the directions it can
go from any position. 

The state transition probabilities are... harder. Dr. Ng models this by saying
`P(forward) = 0.8`, and `P(left) = 0.1` and `P(right) = 0.1`. And it's in fact
very important to model noise in the state transition probabilities, because real
life tends to be pretty noisy. 

And finally, we specify our reward function `R`. In this example, we might reward
our robot +1 by specifically getting to cell `(4,3)`, we might heavily negatively
incentivize the robot for getting to cell `(4,2)`, and we might add a small negative
reward -0.02 for standing in any other space. 

The whole system is going to evolve as follows:
1. Robot wakes up at `s0`. 
1. Based on the state, we choose action `a0`. 
1. The robot is present with a new state `s1 ~ P[sa]`. Note that just because we
    choose an action, doesn't mean we get it's outcome. `P[sa]` often models
    noise. 
1. Choose action `a1`
1. Get to `s2 ~ P[sa]`

And then the total payoff is given by 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return R(s[0]) + gamma*R(s[1]) + gamma**2 * R(s[2])
f

#%% [markdown]
"""
And now we see the point of the discount factor `gamma` - we want the robot to
move efficiently. A reward gained sooner is better than a reward gained farther
down the road, so we want our total reward function to reflect this. Dr. Ng
admits this is shamelessly ripped off of the time-value of money. 

TVoM is a nice story, but it's also just generally true that having a `gamma` 
helps most processes converge much faster. So it's at least as much _ad hoc_ as
it is based on some theory of how rewards should work. 

Mathematically, the goal is to 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return argmin[pi] * E * [R(s[0]) + gamma*R(s[1]) + gamma**2 * R(s[2])]
f

#%% [markdown]
"""
where `pi` is some policy such that `a = pi(s)`, that is, maps from states to 
actions. 

I can't draw the optimal policy, but it essentially is a given direction for
each possible value of being placed on the grid. In particular, I think it's
really fascinating that the policy for (3,1) is leftwards. It's extremely
instructive. It wouldn't make sense if the robot moved with perfect precision,
but since we added the probability of slipping left or right into `P[sa]` -
since we modeled that noise - it's much _safer_ to move towards `(1,1)` then to
try and beeline towards the goal. 

This is the game, though. We try to fit our policy `pi` that maximizes the
reward function, typically by playing the game through millions of iterations and
trying to identify which scenarios maximized our reward the most. It's a neat
way to think about fitting a learning algorithm, and game playing is considered
_the_ primary application of RL. 

### Optimal Policies

In order to build the algorithm for optimal policies, we need to define three terms

$$
V^{\pi}(s) - \textrm{a function that inputs state s and outputs expected total payoff for policy } \pi\\
V^{*}(s) - \textrm{this is the optimal value function, i.e } \max_\pi V^\pi(s)\\
pi^{*} - \textrm{the policy that optimizes} V^{*}
$$

### Belman's equation
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return (V**pi)(s) == R(s[0]) + gamma*Sigma[s[1]] * P[s*pi(s)](s[1]) * (V**pi)(s[1])
f

#%% [markdown]
"""
This says your expected payoff at a given state is the reward at state `s[0]`
plus some discounted expected future reward. That is, just for the fact that
your robot wakes up in state `s[0]`, you get _something_. Could be negative, but
that's the deal. 

And if we have a given $\pi$ and $s$ we can write that expectation as some
indefinite series. 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return (V**pi)(s) == E*[R(s[0]) + gamma*R(s[1]) + gamma**2*R(s[2]) + ... | pi, s[0] == s]
f
#%% [markdown]
"""
And this second form is useful, because if we factor out a $\gamma$, we can 
write the whole thing recursively
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return (V**pi)(s) == E*[R(s[0]) + gamma*(V**pi)(s[1]) | pi, s[0] == s]
f

#%% [markdown]
"""
and state `s[1]` is given by 

$$ s_1 \sim P_{s\pi(s)} $$

so now we've actually proven 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return (V**pi)(s) == R(s[0]) + gamma*Sigma[s[1]] * P[s*pi(s)](s[1]) * (V**pi)(s[1])
f

#%% [markdown]
"""
And this is a linear system of equations in terms of $V^\pi(s)$. In other words,
if you give me policy $\pi$, in our little R2D2 example I can set up a system of
equations with 11 equations and 11 unknowns, one for each value of $s$ in
$V^\pi(s)$. This'll work for any sized discrete policy. 

### Optimal Rewards

There's a way to express $V^*$ in Bellman's Equations
$$
V^*(s) = R(s) + \max_a \gamma \Sigma_{s_1} P_{sa}(s_1)V^*(s_1)
$$

So if you take action a, then the expect output is the immediate reward $R$ plus
the action that maximizes the expected future reward. And let's say we have a
way of computing $V^*$, we can also express
$$
\pi^*(s) = {\arg\max}_a \Sigma_{s_1} P_{sa}(s_1)V^*(s_1)
$$

and this is just the policy that maximizes the future expected reward. 

It all hinges on finding a way to solve $V^*$. Another way to express that $V^*$
_is_ the optimal policy is just

$$ V^*(s) = V^{\pi^*}(s) \ge V^\pi(s) $$ 

Overall, to solve this, we're going to take the following strategy:
1. Find $V^*$
1. Use argmax equation to find $\pi^*$

### Value Iteration

Value iteration goes something like this: 
1. Initialize $V(s) := 0$ for every $s$
1. For every $s$, update:
    > $V(s) := R(s) + \max_a \gamma \Sigma_{s_1} P_{sa}(s_1)V(s_1)$

We can do this synchronously or asynchronously, the difference being that
asynchronos updates are kindof going one at a time, where synchronos is more of
a matrix-style problem that benefits from the efficiencies of matrix notation. 

As we iterate, $V \rightarrow V^*$, and does so fairly quickly. 

As an output, for discrete state spaces, we can actually plot out $V^*$ and see
the expected value at each state. Which is really useful for debugging. Similarly,
the value of any action at any state can be checked as well, which is usually
how you suss out why non-intuitive actions are preferred. 

### Policy Iteration

A different algorithm, with the same goal as Value Iteration. 
1. Initialize $\pi$ randomly
1. Repeat: 
    * Solve Bellman's Equations for $V^\pi$, i.e. $V := V^\pi$
    * Solve for $\pi$
        > $ \pi(s) := {\arg\max}_a \Sigma_{s_1} P_{sa}(s_1)V(s_1) $

Kindof like Expectation Maximization, we solve for $V$ given some value of $\pi$.
That's a linear system of equations. Then we work out a new policy that finds
the expected future value, and bounce back and forth between these two things. 

### Uncertainty in Psa

This is a common modification. We specify these _a priori_ in the R2D2 example, 
but in practice you often don't know these. Chess is a good example. We often
find ourselves having to estimate $P$ from data. So typically the robot is placed
into some kind of random exploration mode where we then estimate
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P[sa] == int((a in s[0]) & s[1] ) / int(a in s[0])
f
#%% [markdown]
"""
directly for all $s$. 

### Putting it All Together

So in total, we'd repeat the following general plan to convergence
1. Take actions w.r.t. $\pi$ to get experience in MDP
1. Update estimates of $P_{sa}$ (and possibly $R$)
1. Solve Bellman's equations using value iteration to get $V$
1. Update $\pi = {\arg\max}_a \Sigma_{s_1} P_{sa}(s_1)V(s_1)$

### Explore

Random exploration is a common way to do our inferences from (1). But it's prone
to locally greedy errors. So our whole procedure may converge on some simple
local optimum, and we don't necessarily want that. So in practice we may
actually inject non-optimal actions into the process more generally. So one
common modification is 
> "Take actions w.r.t. $\pi$" 
turns into 
> "90% of the time we do $\pi$, 10% of the time we choose totally randomly".
This is called "$\epsilon$-greedy exploration". There's a number of algorithms
here, "Boseman exploration" is another we talk about for a bit. There's also
a line of research for "intrinsic motivation RL", where you reward an RL algorithm
for finding something it hasn't found before. 
"""
