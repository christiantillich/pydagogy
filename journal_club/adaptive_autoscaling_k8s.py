#%% [markdown]
"""
# Objective

The goal of [this paper](https://arxiv.org/pdf/2303.03640.pdf) is to introduce a
way of using time-series forecasting to aid the autoscaling system for Alibaba's
Container System on K8s. 

# Background

The paper starts by essentially suggesting that there are three main approaches
to autoscaling kubernetes pods, and that none of them really handle demand 
elastically. The first approach is "don't". That is to say, you have a fixed
number of pods always on and hope it's enough. The other two actually do try and
adjust based on demand - [HPA](https://www.mdpi.com/1424-8220/20/16/4621) and 
[CronHPA](https://www.alibabacloud.com/help/en/container-service-for-kubernetes/latest/cronhpa). 
The authors argue that neither of these really flexibly deal with pod demand - 
HPA _lags_ the demand for pods, and thus makes them available reactively. And
ChronHPA is more or less a pod scheduler - in theory someone could forecast the
demand for pods offline and use that to make some kind of schedule for availability, 
but the process is quite manual and, worst-case, the schedules made may not 
correspond to demand at all. 

Thus, there's a real need for an approach that uses demand forecasting to 
_proactively_ make pods available, based on expected demand. And that is what
the paper argues that it can accomplish. Pretty straight-forward. 

# Methods

The approach taken by the authors here is effectively a traditional time 
series model. 

$$
y_{t+1} = \tau_{t} + \Sigma_{i=1}^m s_{i,t} + r_t
$$

here where $\tau$ is a trend component, $s_i$ the ith seasonal component, and 
r a residual term. They further note that they apply exponential smoothign to 
the trend component and a "quantile regression forest to get an upper bound
prediction of the future residuals". That's an interesting detail that I'd love
to know more about. 

The authors also seem to note that while this above approach is _generally_ 
applicable, they dip into transformer-based deep learning forecasting for

1. Scenarios with enough data
2. Scenarios that need particularly long time horizons. 

And I think it's really interesting here that they paint a picture where they 
are using multiple models here as a kind of system to produce forecasts. I'd
really like to know more about how this is done. 

One thing that this paper opened my eyes to was that it's not enough to simply
forecast the demand - apparently if you're really serious about meeting service
metrics, you want to lean on something called Erlang's C formula. Once you have
the demand, you can work out the probability that the user would wait for a 
pod given the projected demand and the number of pods available. Presumably - 
and I have to admit I'm kindof guessing here - you set the probability to be 
something desired by business and you work backwards through the formula to 
the number of pods you want to make available. 

All in all, it's fairly straightforward. 


# Questions

1. What is RT? e.g. "Different average RT with  different queue models..."
1. I'd be curious to know how often they're rebuilding their forecasts, and other
implementation details like this. While they do mention they don't do much 
hyperparameter tuning, TS models go stale fast. 
1. Makes you wonder what Amazon does. 


"""