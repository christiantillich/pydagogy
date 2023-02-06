#%% [markdown]
"""
# Objective

Discuss Independent Components Analysis. Also discuss cumulative distribution 
functions and the ICA model. After that, we dip into Reinforcement Learning. 

### Independent Component Analysis

The motivating problem is that you have placed a microphone in a crowded room,
and you want to isolate the noise from each speaker independently. He calls this
the "Cocktail Problem". Here we use `i` and `j` a little bit differently, namely

$$s_{j}^{[i]}$$ 

is speaker i at time j, so 
"""
#%% [python]
import latexify
@latexify.expression(use_math_symbols = True)
def f(x):
    return x**[i] == A*s**[i]
f
#%% [markdown]
"""
So the whole goal is to find `W = A**-1` so that 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return s**[i] == w*x**[i]
f
#%% [markdown]
"""
And the rows of `w` we denote with a subscript to correspond to the weights 
required to isolate an single speaker. 

Dr. Ng starts exploring what distribution "voices could take", noting that 
uniform distributions make ICA possible, but Guassian do not. I'm struggling to 
understand how these voices could have any distribution - in other words, it's 
not clear to me how we represent voices in a room to be some data set composed
of components S in the first place. But I'm just gonna hang in here for now and
see where this goes. 

There's, in fact, a theorem that states that if your data is Gaussian, ICA is
not possible, due to the rotational symmetry. Ok, noted. Assuming the data is
non-Gaussian, we derive the ICA algorithm and model. 

We start with the density of `s` and a hint that it's going to be easier to work
backward from the cumulative distribution function. 
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P[s](s) == delta / (delta*s) * P(S <= s)
f

#%% [markdown]
"""
And the goal here is to find the CDF for `S` and work back into PDF `P(s)`. The
first key assumption here is to assume that the CDF is a logistic function
"""

#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return F(s) == P(S <= s) == 1/(1+e**(-s))
f

#%% [markdown]
"""
and the second is to assume that we can partition up the total `P(s)` as the 
product of the individual speakers
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P(s) == Pi[i] * P[s](s[i])
f

#%% [markdown]
"""
Therefore, the density of `x` can be written as 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return P[x](x) == (Pi[j] * P[s](w[j]**T*x))*abs(w)
f

#%% [markdown]
"""
So to do MLE, we set up the likelihood function 
"""
#%% [python]
@latexify.expression(use_math_symbols = True)
def f(x):
    return L(w) == Sigma[i]*log(Pi[j] * P[x](w[j]** T * x**[i]) * abs(w))
f

#%% [markdown]
"""
and then we'll use stochastic gradient descent, using 
$$
\nabla_{w} L(w) = \begin{bmatrix} 1 - 2g(w_{1}^{T}x) \\ ... \\ 1 - 2g(w_{n}^{T}x) \end{bmatrix} x^{[i]T} + (w^{T})^{-1}
$$

as our objective, we can work out the value for `w`. 

To recap, we have some data `x[1] ... x[m]`, and we can use gradient descent to
find `w`, so that by having this map `s = w*x` and working out the derivative
above we can work out source 1 `s[1]`. 

Overall I feel like this section was really rushed, and I'm probably getting a
bunch of things wrong. I really struggle to see how sound data over time actually
fits the mathematical descriptions. I trust that it does, I just don't see it. 
I might try looking elsewhere to bolster my understanding, but I'm also not
sure I'll need ICA in the near future. 

Dr. Ng shows an example of ICA applied on brainwave data. Through ICA, we're able
to isolate specific frequencies that appear to control heart-rate, blinking, etc.
And so ICA becomes a really useful data prep tool, because by isolating these
common default signals, we can then remove them and reconstruct the data without
those components, and this gives a much cleaner view of the EEG data. 

As we get further into the questions, I think Dr. Ng made something kindof explicit
that I think was silently assumed - the technique only really works if you have
as many or more microphones as you do speakers. He says that 1 mic and 2 speakers
is out of scope for the problem, but 5 speakers and 10 mics is totally feasible. 
Most of it centers around the fact that `A` needs to be invertible, but I think
the larger assumption here in the base problem is that every speaker had a mic, 
but each speaker's mic would also pick up other speakers. Whereas I was imagining, 
say, a room full of speakers and maybe like 4 mics. I think this starts to make
sense when we're trying to ask "which mic best represents which speaker?". 

### Reinforcement Learning

Dr. Ng gives an introduction to RL, but I moved that to lecture 17 because the
topic is continued there. 
"""
