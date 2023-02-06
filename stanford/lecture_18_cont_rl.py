#%% [markdown]
"""
# Objective

Continue the Reinforcement Learning discussion, and expand out to continuous
systems. 

### Review

Dr. Ng did a quick review. MDPs are a 5-valued tuple. $V^\pi$ was the value
function for a policy $\pi$, i.e. the expected total payoff for the policy $s$.
$\pi$ itself in a discrete problem is a copy of the state space, with an action
chosen for each. $V^*$ is the _optimal_ value function, which is the absolute
best we're going to do. If we know $V^*$, we can find $\pi^*$, which is just the
set of actions that maximizes the future expected payoff $V$ _given_ the
uncertainty encoded by $P_{sa}$. 

One hint that we're given about things later in the lecture is that we can 
rewrite

$$
\pi^*(s) = {\arg\max}_a E_{s_1 \sim P_{sa}}[V^*(s_1)]
$$

which is probably the trick required to do this in a continuous way. 
"""

#%% [markdown]
"""
### Discretization

Let's start by trying to model out the state space of a car. We assume we've got
a top-down satelite view of the car. We want to know the position $(x,y)$. We
want to know the orientation, say, relative to North $\theta$. We'll also want 
the velocities $(\dot{x}, \dot{y}, \dot{\theta})$. That said, it's kindof
up to you to model the car as well. We could choose to include a lot more features - 
the degree to which the steering wheel is turned, the temperature of the engine, 
the wear on the tires, etc. However, this six-variable state space is good enough
for most normal driving. 

A helicopter might be modeled as $(x,y,z,\phi,\theta,\psi,\dot{x},\dot{y},
\dot{z},\dot{\phi},\dot{\theta},\dot{\psi})$. The inverse pendulum might be 
modeled as $(x,\theta,\dot{x},\dot{\theta})$. There's a lot of room to be as
complex or as simple as the problem needs to be. 

The idea with Discretization is that we're going to take a continuous-state 
problem and just brute-force it to be in discrete space. This usually works
for small problems, but has some disadvantages:  

1. It's a pretty naive representation for $V^*$. Consider bucketing 
    strongly-linear features
1. Curse of dimensionality. Discretization grows as $k^n$ for $k$ dimensions
    and $n$ buckets

But if you've got a simple problem and a small-ish state space, this is a really
good way to start. We have a couple tricks we could employ too. For example, 
if the problem is really sensitive in $x$ but not in $y$, we can make $x$ more
finely grained and $y$ more coarsely grained. 

### Truly Continuous Framings

However, let's assume we want to model $V^*$ without forcing this into discrete
problem. We can take notes from linear regression. Practical regression problems
very rarely use the data totally raw, and instead provide some mapping to a
relevant feature space, e.g. 

$$ y \approx \theta^T \phi(x)$$

where $\phi(x)$ is a mapping of the raw inputs to the features. It can be
higher-ordered dimensions, e.g. $x_1 x_2$, ${x_1}^2$, etc. It could also just be
transformations like missing-value imputation, whatever. Usually $\phi$ is
pretty flexible. Because we have freedom to choose all the relevant features,
and because the whole goal in a continuous RL problem is trying to model the
value of infinitely-fine-grained decision, regression is a natural substitution
for $\Sigma_{s_1} P_{sa}(s_1) V(s_1)$. Thus,


$$ V^*(s) \approx \theta^T \phi(s) $$
"""

#%% [markdown]
"""
### Models/Simulation

In order to derive FVI, Dr. Ng says it's best to use a "simulator". By this, 
he means we need some function that takes as inputs the state and action and
outputs the next state $s_1 \sim P_{sa}$. This is our model. 

Now, it's pretty common for state spaces to be very high dimensional, while your
action space is very low dimensional. So Dr. Ng notes that in a lot of problems
you may still have a discrete action space while having a continuous state
space. So for now we're going to still consider our action space as discrete, but
there'll be hints along the way for how we can modify the problem to deal with 
continuous action spaces too. 

Okay, how do we choose our model? We have a couple options: 

1. There are a number of good off-the-shelf physics simulators. Or we could
    roll our own. Python packages like `gymnasium` just _do_ this for any of the
    games incorporated. It's usually a good first approach. 
1. We could also try and learn it from data. For example, we could place sensors
    on a helicopter and "watch" a human pilot's actions and helicopter state. And
    then we can use some supervised learning approach to make our map, say
    > $s_{t+1} = A s_t + B a_t$
    and fit it using
    > $\min_{A,B} \Sigma_i \Sigma_t || {s_{t+1}}^{(i)} - (A{s_t}^{(i)} + B{a_t}^{(i)})||^2$

Now, an even better choice would be something like  

$$s_{t+1} = A s_t + B a_t + \epsilon_t $$

And usually, you want to model the error. Purely deterministic simulators tend
to produce pretty brittle optimal policies. As far as I can gather, this is 
something we add _ad hoc_ once we have a model, and Dr. Ng says the choice of 
distribution doesn't matter nearly as much as having _some_ error term added. 
"""

#%% [markdown]
"""
### Fitted Value Iteration

1. Sample ${s_1, s_2, ..., s_m}$ randomly
1. Initialize $\theta := 0$ 
1. Repeat:
    1. For $i = 1$ to $m$  
        1. For each action $a$  
            1. Sample $s_1', s_2', ..., s_k' \sim P_{s^{(i)}a} $ (using our simulation)  
            1. Set $q(a) = \dfrac{1}{k} {\Sigma_j}^k [R(s^{(i)} + \gamma V({s_j}')] $ 
        1. Set $ y^{(i)} = {\max}_a q(a)$
    1. Set $\theta := {\arg\min}_{\theta} \dfrac{1}{2} \Sigma_{i} 
        (\theta^T\phi(s^{(i)}) - y^{(i)})$ (i.e. do linear regression $y \sim 
        \theta^T \phi(s) $)

The whole game is to generate a sample $S$ and $y$ so you can set up the
relationship $y \sim S$. However, in practice, there's nothing actually
anchoring us to linear regression here. Any supervised method works. It's "Deep
Reinforcement Learning" if we use a deep NN to estimate $y$

Dr. Ng doesn't explicitly state what actually is the terminal condition for the
repeat step. There seem like two good targets - convergence on `V` or convergence
on `y`. However, once it converges, you have $V^*$

Now, there's no longer an explicit relationship here between $\pi^*$ and $V^*$ 
like there is with the discrete case. I.e. the optimal action may be some 
interpolation of what was observed. So it's no longer trivial to extract the
optimal policy, but it is effectively still

$$ \pi^*(s) = {\arg\max}_a E_{s' \sim P_{sa}}[V^*(s')] $$

A point of clarification, here, is that you really don't want your deployed
policy to have its behavior dependent on a random number generator. We only want
to be injecting noise into the system during training. Therefore, it's fairly 
common practice to set $\epsilon_t$ to 0 in deployment. And then, in deployment, 
you always choose actions to maximize $V(f(s,a))$. 
"""