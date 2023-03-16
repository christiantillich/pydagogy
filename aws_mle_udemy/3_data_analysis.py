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
1. Bernoulli - for binary. 

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

# Quicksight

This is your visualization/dashboarding offering. Lets you build visualizations, 
perform ad-hoc analysis, etc. Can even be done over mobile. Also totally serverless. 

Quicksight works with Redshift, Aurora, RDS, Athena, or any EC2-hosted DB. 

Quicksight works off a platform called SPICE (Super-fast, Parallel, In-memory
Calculation Engine). It'll store the data columnar in-memory, and will take
advantage of this to enable as much parallelization as possible. Each user gets
10GB of storage. 

Main uses are data exploration and the creation of dashboards. There's a couple
new offerings that have an ML-centric focus. The first is anomaly detection, using
AWS' Random-Cut Forest algo. I should play around with this and see if there's
an open source package for this kind of thing. There's also some out-of-box
forecasting offerings, similar to Prophet. And finally, there's a feature called
"Auto-Narratives", which is some kind of AI-driven way of placing dashboards into
a larger script like you might see in a pitch or presentation? I'm a little 
fuzzy on the offerings here. This is it, though, and the teacher makes a point to
say that if there's a question about ML offerings outside of these three, it
must belong to some other product. 

That's a lie - there's actually one more offering. Quicksight Q tries to answer
business questions with NLP. Natural language prompts are input, pretty charts
and graphs that answer the question (allegedly) are output. It's a product that
requires some serious training, as Q has to be told where to find the data that
it's going to be compiling, as well as how it's all supposed to be formatted. 

What do we _not_ use QuickSight for? There's a controversial opinion on whether
the test might expect us to know about paginated reports. They were released
Nov. 2022. It _used_ to be the case that QuickSight was more about
rough-and-ready dashboards, and less about managerial quality paginated reports,
but since Nov, this is no longer the case. However, the test may still not
account for this yet. So it's a sticky question. I'd probably err on the side
of the test being up to date, though. 

We also don't use quicksight for ETL, that's more Glue's function, even though
Quicksight does have some limited transformative capability. 

Security: QS supports MFA, VPC connectivity, and row/column security. You also 
get charged per user, so there's some user management defined using IAM.

They actually charge for a lot more, very nickle-and-dimey, pay more for Q, pay
more for extra SPICE, pay more for paginated reports, anomaly detection, readers, 
etc. 

### Visualization Types

There's going to be several questions on "given this data, how should we present
it?" So buckle up. 

AutoGraph will do the visualization stuff automatically, though it's more error
prone. Other visual offerings include:  
* Bar
* Line Graphs
* Scatter Plots
* Heat Maps
* Pie Graphs
* Tree Maps
* Pivot Tables
* Geospacial
* Donut Charts
* Gauge Charts
* Word Clouds 

We know what these all do, so I'm not gonna spend much time on it. The only 
thing to call out is that "Tree Maps" might be considered to be a good way of 
displaying hierarchical information. I would _profoundly disagree_, but play to
the audience. 

# Elastic Map Reduce

This is a managed Hadoop framework, offered up and often including Spark, HBase,
Presto, Flink, Hive, and some other frameworks that might also sit on top of
Hadoop and provide additional functionality. This is for absolutely massive data
sets, where you need a whole cluster of computers to evaluate or process. 

EMR clusters are typically composed of Master Nodes, Core Nodes, and Task Nodes. 
A Master Node's job is to control the cluster and orchestrate, it's a single 
EC2 Instance responsible for all the rest. Core Nodes host data. They can be
scaled up or down, but you risk losing data when you do. The task notes are just
there to run tasks, and don't actually host the data, so these are the most
scalable of the nodes. Thus, they make the best fit for when you need spot
instances. 

Generally, we have two use cases in mind when spinning up a cluster. Transient
clusters have an expiration date, usually after some set number of tasks have 
been achieved. Long-Running Clusters, in contrast, are meant to host 
applications indefinitely. 

EMR makes use of the following services:  
* EC2 - These will make up the individual nodes. 
* VPC - Does the networking for the cluster. 
* S3 - Stores your input/output data. 
* IAM - for permissions
* CloudWatch - For performance and alerting. 
* CloudTrail - Request Auditing
* Data Pipeline - Scheduling the cluster on/off for specific tasks. 

There's a couple different storage options here. We can run HDFS, which stores
the data directly on the cluster. However, as soon as you delete the cluster, 
you'll lose that storage. But the tradeoff is speed, things will go much faster
if the data is all available locally on the machines doing the processing, and 
Hadoop has a lot of intelligent optimization that will make the query process
the fastest. So that's the landscape. 

EMRFS is something of a split-the-difference option instead. The data is stored
on S3, but EMRFS gives a layer that makes it behave like HDFS. 

Finally, Amazon does offer Elastic Block Store, which seems to be a custom
storage offering specifically for Hadoop and EMR. 

Pricing is per-hour for the EMR service, plus all the EC2 time used by the 
cluster. I'm guessing that adds up fast. But you get automatic provisioning in
failure, as well as all the benefits of clustered computing. 

### Spark on EMR

Before we get to Spark, we need to think about Hadoop as three comoponents all
sitting on top of each other - HDFS | YARN | MapReduce. HDFS is the file
management system, YARN negotiates resources for the cluster, and MapReduce is
the transformation element that's setting up chains of map-reduce to get the
desired output. 

This structure is important because Spark pushes out and replaces the MapReduce
component, and is widely considered superior to MapReduce. Actual work is done
in the "Spark Context" in the master node, which talks to the cluster manager
(YARN, but there's also a spark component for non-hadoop clusters) to decide how
to best divy up resources. 

There's a couple other Spark components here - Streaming, SQL, MLLib, GraphX. 
The SQL component is pretty robust, interacting with many common file types
and even pushing out the Core dataframe components for its own dataframe object
that many consider superior. It's also very similar to a Pandas dataframe in 
terms of look and feel, but one that can distribute the data and specified 
computations across the whole cluster. There's a streaming component as well, 
which integrates well with Kafka and Kinesis and allows users to define how 
to handle a batch, which then gets applied on Spark Streaming "mini-batches"
for streamed analytics. The MLLib is the ML library, and is a collection of 
model implementations designed to take advantage of the distributed computing
power. And GraphX is similarly your network processing library - i.e. edges and
nodes, not visualization. 

With MLLib, it's important to note that Spark only really cares to implement 
algorithms that can take advantage of the parallel processing in some way. So 
the offerings are kindof weird. Some classification and regression, tree methods, 
clustering, LDA, SVD/PCA, etc. I must confess I'll have to dig in on some of these
to understand why they benefit from a parallel computation structure. 

With spark streaming, imagine having a data table with no rows to start, and 
where rows get appended over time. This is essentially how Spark treats data
streams. With spark, you can define a structured stream using Kinesis as an 
input, as a way to query out the results of the stream. 

Spark itself has a notebook option called Zeppelin (is this a Hindenberg joke?). 
It gives you a spark shell, a SQL executor, and some light visualization
for building dashboards and analyses. 

### EMR Notebooks

AWS gives a counter-offering here to Zeppelin. Notebooks get backed up to S3, 
you can provision cluster resources directly in the notebook, and it can be 
accessed right from the AWS console. 





"""
