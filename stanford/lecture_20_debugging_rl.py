#%% [markdown]
"""
# Objective

Last lecture, he's got jokes. We're going to outline main debugging/diagnostic
strategies. 
"""

#%% [markdown]
"""
### RL debugging/diagnostic

Suppose the following three statements are true in the helicopter flight 
simulation. 

1. The simulator is accurate. 
2. The RL algorithm correctly controls the helicopter so as to maximize the
    expected payoff $V^{\pi}$
3. Maximizing expected payoff corresponds to correct autonomous flight. 

Then the learned controller $\pi$ should fly well on the actual helicopter. 
Therefore, if $\pi_{RL}$ is performing poorly, it must be because one of these
expressions isn't true. We can ultimately check the problem, then, by asking
the following questions. 

1. Does $\pi_{RL}$ work in simulation? If so, we probably need a more advanced
    simulator. 
1. Let $\pi_{human}$ be the human control policy. If $V^{\pi_{RL}}(s_0) < 
    V^{\pi_{human}}(s_0)$, the problem is the reinforcement algorithm. It's not
    finding the true max payoff possible. 
1. Otherwise, the problem is the cost function. There's a niche cache of very
    high rewards that the RL algo finds that doesn't translate to flying well. 

Dr. Ng talks for a bit how the bottlenecks move around on a project like this. 
For the first couple months, the simulator is the bottleneck - it's just not
good enough. After a while, (1) stops being the problem, and maybe the new
bottleneck becomes the reward function. So you spend some time on that, and then
the next bottleneck appears. You might get to (3), but some little niche case
unearths an error in (1), so you have to go back. But in general, teams that
prioritize their diagnostics in this framework operate much more efficiently. 
So, something to keep in mind. 

We go into the case of what looks to be one of the Boston Dynamics robot dogs. 
The trick seemed to be to get some kind of video of the terrain, and the height
of the terrain, and the RL algo would learn the safe place to put its feet. In
other words, feet placed on edges stood a much greater chance of slipping. 
"""



#%% [markdown]
"""
### Direct Policy Search & POMDPs

"Policy Search" is a type of algorithm that seems to try and find optimal values
of $\pi$ directly, as opposed to figuring out $V^*$ first and then using that 
function to mathematically determine $\pi$. 

In logistic regression, we kindof start by assuming that our function has a 
fixed form like 

$$ y \approx h_{\theta}(x) = \dfrac{1}{1+e^{-\theta^Tx}} $$

We _could_ assume something similar about $\pi$. 

$$ a = \pi_{\theta}(s) = \dfrac{1}{1+e^{-\theta^Ts}} $$

And so we define 
> __stochastic_policy__: a function $\pi(s,a)$ giving the probability of taking 
> action $a$ in state $s$

So for our inverted pendulum problem, we might frame the problem as 

$$ 
\pi_{\theta}(s, a = "right") = \dfrac{1}{1+e^{-\theta^Ts}} \\
\pi_{\theta}(s, a = "left") = 1 - \dfrac{1}{1+e^{-\theta^Ts}}
$$

And we've we express the params as 

$$ 
\newcommand\p[1]{\begin{pmatrix}#1\end{pmatrix}}
\newcommand\b[1]{\begin{bmatrix}#1\end{bmatrix}}
s = \p{1 \\ x \\ \dot{x} \\ \phi \\ \dot{\phi}} \\ 
$$

so that an expression of 

$$ \theta = \b{0 \\ 0 \\ 0 \\ 1 \\ 0}$$ 

would result in $\pi(s,a="right") = \dfrac{1}{1+e^{-\phi}}$, meaning that whenever
the pendulum tips to the right, we move right to try and get under it. The goal, 
in this framing, is to find a policy that dictates theta dependent on $s$
> __Goal__: Find $\theta$ so that when we execute $\pi_{theta}(s,a)$, we maximimze
> $ \max_{\theta} E[\Sigma_t R(s_t,a_t)|\pi_{\theta}] $

Note that here, $s_0$ is a fixed initial state. This is a little different than
previous problems, which were independent of starting state. But it's an 
assumption that's pretty common in practice. So let's write it out and start
filling in the blanks. Let's work with just a short summation for now. 

$$ 
\newcommand\params{(s_0,a_0,s_1,a_1)}
\newcommand\payoff{R(s_0, a_0) + R(s_1, a_1)}
\max_{\theta} E[\Sigma_t R(s_t,a_t)|\pi_{\theta}] \\
\max_{\theta} \Sigma_{\params} P\params [\payoff] \\
\max_{\theta} \Sigma_{\params} P(s_0) \pi_{\theta}(s_0,a_0) P_{s_0a_0}(s_1)\pi_{\theta}(s_1,a_1)[\payoff]
$$

And so we can come up with a framing of gradient descent for this big ugly 
guy above. This algo is called the "Reinforce Algorithm". There's some implementation
details, but it goes like this 
$$
\newcommand\pol[1]{\pi_\theta(s_{#1},a_{#1})} 
\newcommand\grad[1]{\dfrac{\nabla_{\theta}\pol{#1}}{\pol{#1}}} 
\newcommand\payoff{R(s_0, a_0) + R(s_1, a_1) + ...}
\newcommand\payoffsimp{R(s_0, a_0) + R(s_1, a_1)}
$$

1. Loop
    1. Sample $s_0, a_0, s_1, a_1, ...$
    1. Compute the payoff $R(s_0) + R(s_1) + ...$
    1. Update $\theta := \theta + \alpha(\grad{0} + \grad{1} + ...)[\payoff]$

And it turns out that updating $\theta$ in this way isn't actually stochastic, 
or at least the average update is in the exact direction of the gradient. The 
state sample is random, but there's kindof a wisdom of the crowds thing happening
here where the expectation of the grads always points in the upward direction. 
So it's _less_ stochastic than SGD. And we can show this, back to 2-term for simplicity

$$ \nabla_{\theta} \max_{\theta} \Sigma_{\params} P(s_0) \pi_{\theta}(s_0,a_0) P_{s_0a_0}(s_1)\pi_{\theta}(s_1,a_1)[\payoffsimp] $$

Let's focus on this first term for a second. By the product rule

$$ \Sigma_{\params} (P(s_0) (\nabla_{\theta}\pol{0})P_{s_0a_0}(s_1)\pol{1}+ P(s_0)\pol{0}P_{s_0a_0}(s_1)\nabla_{\theta}\pol{1}) $$

And with some clever choice of denominators

$$ 
\newcommand\dpol[1]{\dfrac{\pol{#1}\nabla_{\theta}\pol{#1}}{\pol{#1}}}
P(s_0)\dpol{0}P_{s_0a_0}(s_1)\pol{1} + P(s_0)\pol{0}P_{s_0a_0}(s_1)(1)\dpol{1}
$$

We can factor out the probability of the whole state sequence

$$ P\params = P(s_0)\pol{0}P_{s_0a_0}\pol{1} $$

So that

$$ \Sigma P\params [\dfrac{\nabla_{\theta}\pol{0}}{\pol{0}} + \dfrac{\nabla_{\theta}\pol{1}}{\pol{1}}][\payoffsimp] $$

Neat, but when do we use Direct Policy Search? There seem to be a couple great
cases. The first is Partially Observable MDPs (POMDP). In the inverted pendulum
case, let's say we can see the position and angle via sensors, but we don't
have a velocity sensor. So we have a partial (and potentially noisy) measurement
of the state of the system, and still have to choose actions $a$. This is a great
case for Direct Policy Search. 

Let's say on every time step you observe

$$ y = \p{x \\ \phi} + \epsilon $$

in our pendulum problem. The policy

$$ \pi(y, a="right") = \dfrac{1}{1+e^{-\theta^Ty}}  , y = \p{1 \\ x + \epsilon \\ \phi+\epsilon} $$

works extremely well, and we've cut out half the information
"""


#%% [markdown]
"""
### Final Thoughts

Reinforcement Learning is actually pretty inefficient. Fairly common to see 
people run millions of iterations - long times and small learning rates. But 
the right way to think about RL learning as a whole is to start thinking about
which is simpler, $pi^*$ or $V^*$. 

Dr. Ng thinks about this as low-level vs high-level control tasks. Low level
control is instinctual, seat of the pants decision making like flying a 
chopper or driving a car. Things like playing chess are high-level control tasks, 
and he frames this more as multi-step, looking-into-the-future style reasoning. 
Driving the car normally is low-level, but aggressive maneuvers like trying
to pass one car while overtaking the other car and avoiding a pedestrian, that
is more high-level control. 

Low-level control tend to have simpler $pi^*$. High level control tends to find
it easier to find $V^*$. Thus low level control might be easier to frame as 
Direct Policy Search, whereas high-level as Value Policy Iteration. 

RL is killing it in game playing and robotics right now. Also chatbots and 
guidance counselors, and so on. Some applications in medical planning, too, where
we might be concerned about a couple diagnoses, and we want an efficient path
of testing to take to figure out what the issue is. 
"""
