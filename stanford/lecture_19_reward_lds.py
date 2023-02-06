#%% [markdown]
"""
# Objective

Last week of the class. We're going to go over some generalizations of MDPs: 
State action rewards and finite-horizon MDPs. Then we'll talk about linear
dynamical systems. 
"""

#%% [markdown]
"""
### State Action Rewards

In a normal MDP, we deal with a system where the reward is mapped 1-1 to the
state. SARs generalize this framework a little so that the reward is a function
of both the state and the action

$$ R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + ...$$

In this instance, Bellman's equations becomes 

$$ V^*(s) = \max_a [R(s,a) + \gamma \Sigma_{s1} P_{sa}(s') V^*(s')] $$

We're still fundamentally dealing with the same equation, only now the immediate
reward is now dependent on the action as well, so the $\max$ applies to the whole
thing. And the optimal policy changes to 

$$ \pi^*(s) =  {\arg\max}_a [R(s,a) + \gamma \Sigma_{s1} P_{sa}(s') V^*(s')] $$
"""

#%% [markdown]
"""
### Finite Horizon MDP

In a Finite Horizon MDP, we replace $\gamma$ with a horizon time $T$. We've mostly
looked at discrete cases, and that kindof _seems_ like finite time, but it actually
hasn't been. Those systems would run infinitely, and the robot would have just
stayed and collected $1$ indefinitely. 

Finite Horizon MDP gives you a finite amount of time - e.g. fuel in a helicopter,
time on the shot-clock, etc. The interesting effect of this is that the optimal
action can get more "desperate" towards the end, taking riskier actions or 
being more "greedy" in general. 

So we tend to write the optimal policy now as ${\pi^*}_t(s)$. This is technically
referred to as a non-stationary policy, and there's different flavors too. We
can go for non-stationary state transitions 

$$ s_{t+1} \sim P_{s_ta_t}^{(t)} $$

We can have rewards that vary over $t$

$$ R^{(t)} (s,a) $$

Dr. Ng gives the example of weather forecasts and industrial labor as examples
where time-of-day really has an impact on reward and state transition
probabilities. However, he then goes to ignore the superscript to deal with 
the subproblem of 

$$ V^*_t(s) = E[R(s_t, a_t) + R(s_{t+1}, a_{t+1}) + ... + R(s_T, a_T) | \pi^*, s_0 = s]$$ 

But this becomes a dynamic programming problem now, because 

$$ 
V^*_T(s) = \max_a R(s,a) \\
V^*_t(s) = \max_a [R(s,a) + \Sigma_{s'} P_{sa}(s') V^*_{t+1}(s')] \\
\pi^*(s) = {\arg\max}_a [R(s,a) + \Sigma_{s'} P_{sa}(s') V^*_{t+1}(s')] 
$$ 

The issue is, now, that the model has to know where it's going to be at time $T$,
figure out the optimal action there, and then work its way backwards. It _seems_
like this is the first case we're doing this, though to be honest it's not clear
to me why we _weren't_ doing this in Fitted Value Iteration. But that's probably
going to become clearer as I get more hands-on experience with RL and all the
different flavors of it. 


"""

#%% [markdown]
"""
### Linear Dynamical Systems

It's easier to do this in the finite-horizon setting, so we're going to stick
with that for now. We've got our MDP of $(S, A, P_{sa}, T, R)$ and we're going
to assume that the future state evolves according to 

$$ s_{t+1} = \mathcal{A}s_t + \mathcal{B}a_t + w_t $$

Here, $\mathcal{A}$ and $\mathcal{B}$ are both matrices that essentially just
map the current states and current actions to the future state. $w_t$ here is an
error term, and it's not really going to matter too much, but we'll assume it's
Gaussian. 

We also make one final assumption about the reward, namely

$$ R(s,a) = -(s^T \mathcal{U} s + a^T \mathcal{V} a) $$

and both $\mathcal{U}$ and $\mathcal{V}$ are positive semi-definite. Note, too, 
that if we assume $\mathcal{U}$ and $\mathcal{V}$ are both identities, the 
reward function simplifies to 

$$ R(s,a) = (||s||^2 + ||a||^2) $$

The goal now becomes finding a solution for $\mathcal{A}$ and $\mathcal{B}$. We
can frame our cost function as 

$$ \min_{\mathcal{A},\mathcal{B}} \Sigma_i \Sigma_t || s_{t+1} - (\mathcal{A}s_t + \mathcal{B}a_t ) ||^2 $$

We have a couple ways of solving this  
1. We could try and learn the parameters from pilot data. 
1. We can try and linearize the system. 

Dr. Ng doesn't really elaborate on (1), so I'm not sure how we do that, but 
goes on to talk about (2). What we have, essentially, is a function

$$ 
\newcommand{\state}[1]{\begin{pmatrix} x_{#1} \\ \dot{x}_{#1} \\ \theta_{#1} \\ \dot{\theta}_{#1} \end{pmatrix}}
\state{t+1} = f(\state{t}, a_t) 
$$

that models the current state and current action. The idea behind linearization
is that we find the tangent at some constant $\bar{s}_t$ 

$$ s_{t+1} \approx f'(\bar{s}_t) * (s_t - \bar{s_t}) + f(\bar{s}_t) $$

and use that tangent line to approximate the real behavior (This is Taylor's
theorem, yes?). The generalization to a function $f(s,a)$ goes something like

$$
\newcommand\func{f(\bar{s}_t, \bar{a}_t)}
\newcommand\cent[1]{(#1 - \bar{#1}_t)}
s_{t+1} \approx \func + (\nabla_s \func)^T\cent{s} + (\nabla_a \func)^T \cent{a}
$$

In english, the next state is approximately linear given some starting point plus
the linear rate of change in $s$ times the amount you want to move in $s$ plus
the linear rate of change in $a$ times the amount you want to move in $a$. This
is what we mean by "linearizing". 

### LQR

So where are we? The problem we're trying to solve is finding $\mathcal{A}$ and
$\mathcal{B}$ in order to model state transitions as

$$ s_{t+1} = \mathcal{A}s_t + \mathcal{B}a_t + w_t $$

and we assume the reward is 

$$ R(s,a) = -(s^T \mathcal{U} s + a^T \mathcal{V} a) $$

with a total payoff

$$ \Sigma_t R(s_t, a_t) $$ 

for finite $T$. The trick here we're relying on is that if we accept all of the
above assumptions, $V^*$ turns out to be quadratic. And if $V^*$ is quadratic, 
it has a closed form solution. We start by expressing the optimal value at the
last step

$$
V^*_T(s_T) = \max_{a_T} R(s_T, a_T) \\ 
V^*_T(s_T) = \max_{a_T} -(s^T_T \mathcal{U} s_T + a^T_T \mathcal{V} a_T) \\
V^*_T(s_T) = -(s^T_T \mathcal{U} s_T)
$$

meaning that our last action adds the most value by being zero always, since
any additional action would lower our value function. Thus

$$ \pi^*_T(s_T) = 0 $$

the best final policy is to do nothing. We can work with that. 

So the last step is to express $V^*$ as some quadratic function 

$$ V^*_{t+1}(s_{t+1}) = s^T_{t+1} \Phi_{t+1} s_{t+1} + \Psi_{t_+1} $$

and we want to find $V^*_{t+1} \rightarrow V^*_t$. So our dynamic programming 
step is 

$$ V^*_t(s_t) = \max_{a_t} [R(s_t, a_t) + E_{s_{t+1} \sim P_{s_t a_t}}[V^*_{t+1}(s_{t+1})]]$$

Once again, the flavor here is to choose an action to maximize the initial payoff
plus some expected future payoff. 

$$
\newcommand\immed{-s_t\mathcal{U}s_t - a_t\mathcal{V}a_t}
\newcommand\dist{s_{t+1} \sim \mathcal{N}(As_t + Ba_t, \sigma)}
\newcommand\future{s^T_{t+1} \Phi_{t+1} s_{t+1} + \Psi_{t_+1}}
V^*_t(s_t) = \max_{a_t} [\immed + E_{\dist}[\future]]
$$

This does get even hairier, but essentially our expectation can be written
as just a quadratic function of $a_t$. One that we can take the derivative of
and solve for $a_t$. We don't go through that here. But the solution is 

$$ 
\newcommand\m[1]{\mathcal{#1}}
\newcommand\L{(\m{B}^T \Phi_{t+1} \m{B} - \m{V})^{-1} \m{B}\Phi_{t+1}A}
a_t = \L s_t \\ 
a_t = L_t s_t
$$

All of this effort serves to get us to a point where the optimal action is 
just a linear function of the state. And thus the optimal policy also a linear
function of state

$$ \pi^*_t(s_t) = L_t s_t$$

And with a _lot_ of algebraic massage, we may rewrite $V^*_t$ as 

$$
V^*_t(s_t) = s_t^T \Phi_t s_t + \Psi_t \\
\Phi_t = f(\Phi_{t+1}, \m{A}, \m{B}, \m{U}, \m{V}) \\ 
\Psi_t = f(\Phi_{t+1}, \Psi_{t+1}, cov(w_t))
$$

Dr. Ng actually writes out the actual formulas, but because of some ambiguity in
notation I'm not actually sure I can parse fully what they are. But I don't 
think it's terribly relevant - we have $\Phi$ and $\Psi$ as functions of a later
time, plus a bunch of matrices that we work out the values for. We have an 
expression for $V^*$ and $\pi^*$ at the very end. And we can ultimately work
backwards from the end point iteratively to the optimal strategy. The full
algorithm goes like 

1. Initialize $\Phi_t = -\m{U}$ and $\Phi_T = 0$
1. Recursively calculate $\Phi_t, \Psi_t$ using $\Phi_{t+1}, \Psi_{t+1}$
1. Calculate $L_t$
1. $\pi^*(s_t) = L_t s_t$

The neat thing about this is that while we effectively approximate our state 
transitions to be locally linear, we calculate exactly everything else. And since
we typically frame our RL systems as taking small actions in small increments of
time, this works out really well. 

One neat thing about this LQR algorithm is that $pi^*$ doesn't actually depend
on the noise in the system, and you can do the whole thing without ever _touching_
$\Psi$. This is a neat property of LQR only, and doesn't really generalize, but
if you can torture your RL problem _into_ an LQR problem, we can ignore the 
noise altogether. 
"""