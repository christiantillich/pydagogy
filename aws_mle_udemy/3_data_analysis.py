#%% [markdown]
"""
# Objective

Review the AWS offerings for Data Analysis. This is mostly a course on variable
transformation. 

# Python and Jupyter

For the AWS test, we just kindof need to know the basic packages available. 
Pandas, numpy, maplotlib (but plotly is better), scipy, scikit, etc. The 
teacher assures us several times that we don't actually need to know python
for the exam, but that instead we need to understand things at a high level. 

We go over an example notebook from the UCI repository that takes in images of
mammogram screenings and tries to predict cancer, along with some other relevant
patient info - e.g. age, shape of mass, density of mass, etc. 

# Types of Data

* Numerical - comes in discrete and continuous. 
    * Discrete data can further be subvided into integers and count data. 
    * Continuous - infinite precision. 
* Qualitative - Binary flags and purely categorical values. 
* Ordinal - like categorical, but order matters. Likert scales, 1-5 stars, etc.

# Distributions

1. Normal - we all know it, we all love it. Remember you can't specify the prob
    of an exact value, so we do everything off ranges e.g. $P(x > 2 \sigma)$ 
1. Binomial Approximation of Normal
1. Poisson - for count data where every obs has the same avg. value. 
1. 

There's something kindof weird here going on for the test. The teacher doesn't
really cover why you would use them, just really emphasizes that if they're
mentioned, we can use them to infer what type of data we're dealing with. 

# Time Series

Typical decomposition into trend, seasonality, and error. Shows an example of 
denoising by extracting the seasonal components. Also explains that the test
might cover the difference between additive or multiplicative models. 

# Athena

Interactive query service for S3. No need to load data into anything - it's fully
serverless. One example used is you've got a bunch of web logs just written out to 
S3. Athena's capable of running queries against. 

Glue is something of an enabler here - Glue crawlers can extract out the metadata
information, which Athena can then use to do the actual data searching. 

Athena appears to position itself as a very low-barrier-to-entry DB system. It's
not Redshift, it's not RDS, it's something to query raw files that are just 
dumped into S3. 





"""
