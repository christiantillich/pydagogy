¢#%% [markdown]
"""
# Objective

Review S3 as a service and our main data storage tool. 

### Overview

S3 allows people to store "objects" (files) in "buckets" (directories). The
buckets must have a globally unique name, and the object's key is the full 
path `<bucket>/file_path.txt`. The max object size is 5TB, but S3 also supports
partitioning. Also, object tags are a thing we talk about later. 

This tool is the backbone for AWS ML services, generally. S3 is totally decoupled
from the compute side, allowing you to kindof take it anywhere. Furthermore, 
It supports _any_ file format. 

### Partitioning

Refers to breaking up S3 paths into easy-to-search chunks, for example 
`s3://bucket/my-data-set/year/month/day/hour/data_00.csv`. 

If we're querying by date (for example with AWS Athena) this setup allows us
to very quickly pull down the data within a range. However, we're only going to 
get 1 partitioning strategy, so it may be more beneficial to do something like
`s3://bucket/my-data-set/product-id/data_32.csv` instead. In this case, we've
optimized our data retrieval by product instead. 

Some AWS tools have a partioning strategy that they use already (like AWS Glue).
Something to keep in mind. 

### Python Interface

`Boto3` is a pretty comprehensive API to the AWS ecosystem. We go through an
example of uploading data in S3 through the web portal, but we could access it
in python by creating a Boto S3 client and asking it to list objects. 
"""

#%% [python]
import boto3 as aws
import pandas as pd
import io
s3 = aws.client('s3')

#%% [markdown]
"""
There are a number of options, but these are some of the most important to us. 
We can list out the buckets. 
"""

#%% [python]
pd.DataFrame(s3.list_buckets()['Buckets'])

#%% [markdown]
"""
And we can list out the files that are within a bucket.
"""

#%% [python]
pd.DataFrame(s3.list_objects(Bucket="nmr-sandbox")['Contents']) 

#%% [markdown]
"""
Note that we've decided to store our files in the path `ctillich/s3_tutor`, so 
if we want to retrive a truncated list, we might do something like 
"""
#%% [python]
pd.DataFrame(s3.list_objects(Bucket="nmr-sandbox", Prefix = "ctillich/s3_tutor")['Contents']) 

#%% [markdown]
"""
And finally, we'll want to retrieve the object. This is a little bit convoluted
using Boto
"""
#%% [python]
pd.read_csv(io.BytesIO(
    s3
        .get_object(Bucket="nmr-sandbox", Key = "ctillich/s3_tutor/2022/10/21/instructor-data.csv")
        ['Body']
        .read()
))

#%% [markdown]
"""
### s3fs

Another option available to us is s3fs. If we have s3fs imported, pandas can 
read csvs from it with no extra code. 
"""

#%% [python]
import s3fs
pd.read_csv("s3://nmr-sandbox/ctillich/s3_tutor/2022/10/21/instructor-data.csv")

#%% [markdown]
"""
Using s3fs natively, this is how we might list and read files. 
"""

#%% [python]
fs = s3fs.S3FileSystem()
fs.ls('nmr-sandbox/ctillich/s3_tutor/2022/10/21/')   

#%% [python]
with fs.open('nmr-sandbox/ctillich/s3_tutor/2022/10/21/instructor-data.csv') as f:
    df = pd.read_csv(f)
df
#%% [markdown]
"""
And there's other simplified methods for more complex operations, like downloading
or uploading, etc. 

### S3 Storage Classes and Glacier

S3 has a number of different storage classes: 
* General Purpose
* Standard-Infrequent Access
* One Zone-Infrequent Access
* Glacier Instant Retrieval
* Glacier Flexible Retrieval
* Glacier Deep Archive
* Intelligent Tiering

When creating s3 objects, we can choose the class manually. We can also set up
S3 lifecycle configurations to handle this for us as well. 

AWS is actually fairly open about the fact that storing objects can sometimes 
result in losing them, which I guess is neat, and they quantify this with a 
measure called _durability_.  

> __Durability__: The probability of losing an object. 

AWS quotes 99.999999999% durability, meaning if you store 10 million objects
with S3, you can on average expect to incur a loss of a single object once 
every 10,000 years. Pretty...    pretty durable, yeah. This durability is the
same for all storage classes. 

But just because an object is there doesn't mean it's accessible. This is handled
by _availability_. 

> _Availability_: The percentage of time (over a year?) that your objects will be
> inaccessible. 

This _does_ vary by storage class. 
"""

#%% [python]
classes = {
    "type": [
        "General Purpose", "Infrequent Access", "One-Zone Infrequent", 
        "Glacier Instant", "Glacier Flex (Exp)", "Glacier Flex (Std)", 
        "Glacier Flex (Bulk)", "Glacier Deep (Std)", "Glacier Deep (Bulk)"
    ]
    ,"availability": [
        "99.99%", "99.9%", "99.5%", "99.99%", "99.99%", "99.99%", "99.99%", 
        "99.99%", "99.99%"
    ]
    ,"retrieval_duration": [
        "inst","inst","inst", "~ms","1-5 mins","3-5 hrs", "5-12 hrs", "12 hrs",
        "48 hrs"
    ]
    ,"use_case": [
        "Like a Hard Drive", "Disaster Recovery",
        "On premise backup - lost if zone is destroyed", 
        "Backups where acceses requires millisecond availability - e.g. quarterly reports", 
        "Longer Backups (tiered)","Longer Backups (tiered)",
        "Longer Backups (tiered)", "Longest Backups" , "Longest Backups"
    ]
    ,"facility_failure_count": [2, 2, 1, 2, 2, 2, 2, 2, 2]
    ,"min_storage_duration": [
        pd.NA, "30 days", "30 days", "90 days", "90 days", "90 days", "90 days", 
        "180 days", "180 days"
    ]
}
pd.DataFrame(classes)

#%% [markdown]
"""
There's also an option called Intelligent Tiering, where you pay a monthly fee
and AWS moves objects between access tiers _for you_. This removes retrieval 
charges. It _mostly_ seems comparable to GP access in terms of features, but
instead of pricing by amount of data retrieved, you're priced per month by the
number of objects monitored. 

We go into a hands-on lab changing the storage classes of a .jpg object through
the AWS console. For reference, I saved it at
"""
#%% [python]
fs.ls('nmr-sandbox/ctillich/storage-classes-demos')   

#%% [markdown]
"""
And we _should_ be able to see and even manipulate the storage class using `s3fs`, 
but in practice it doesn't seem to work. 
"""
#%% [python]
fs.getxattr('nmr-sandbox/ctillich/storage-classes-demos/coffee.jpg', 'x-amz-storage-class') 

#%% [markdown]
"""
Boto might also have something here, but I'm honestly probably not going to be
manipulating this in python all that often. 

### Lifecycle Rules 

Lifecycle Rules are ways of handling the storage class of objects automatically.
In the AWS console, this is under the "Management" tab of any selected bucket.
You can essentially set `if/then` statements based on object creation or the
object version change date. You can set lifestyle rules for a whole bucket, or a
specific path too. 

S3 versioning also allows it so that deleted objects are just a new "version"
tagged for "delete". Enabling versioning also allows us to set lifecycle rules
based on this status, so if the business problem is 

> I want users to be able to recover deleted objects immediately up to 30 days >
after deletion, and within 48 hours therafter

The solution is to set lifecycle rules on all non-current versions of the
object. The most cost-effective approach is to set non-current versions to
Standard IA, since we expect to have to recover deleted objects rarely, and then
after 30 days transition non-current versions to Glacier Deep Archive. 

AWS also has a report called "Storage Class Analysis" for objects stored as
Standard or Standard IA. You can use this report to see objects, their current
class, and their ages. It's what you want if you're trying to create or modify
lifecycle rules as a cost-saving measure. 

There was a lab, I didn't follow along, because I don't own the buckets I'm
using. But Stephane sets up lifecycle rules for the bucket, transitioning to 4
different storage classes at 30, 60, 90, and 180 days. All fairly
straightforward to do off the Management tab. 

### Security

There are four methods of encryption supported

1. SSE-S3 - AWS handles it all. 
2. SSE-KMS - Use AWS Key Management Service to manage encryption keys. 
    1. Gives us a way to gatekeep users through KMS access
    1. Gives us an audit trail for key usage. 
3. SSE-C - I manage the encryption keys internally. 
4. Client Side Encryption - I send data to some 3rd party for encryption before
   sending to AWS

We're probably mostly going to use S3 and KMS. The others are probably for more
sensitive data than anything I'm doing. 

In addition, we can set S3 access permissions in two ways. 

1. User based control - through IAM
1. bucket/object based control
    1. Bucket policies
    1. Object Access Control Lists
    1. Bucket Access Control Lists

Bucket policies are json documents. The docks specify which resources are
controlled, which API actions are allowed by setting "Allow" or "Deny", and
which accounts the policy should apply to. 

So we could use a bucket policy to grant public access, force encryption on
upload, and grant access to other accounts, for instance. 

This is all probably a bit in the weeds, but mostly it looks like I can set
encryption through the S3 portal by clicking on objects and going to the
"Properties" tab. We can do this similarly as a policy by viewing the properties
of the bucket, but only for new objects uploaded. 

Other things to know:

* You can set a VPC Endpoint Gateway, which means that instead of making
  requests for bjects over the wide internet, you can first log into a virtual
  private cloud and handle all S3 requests from there. I suspect this is how
  things are done in most businesses. 
* S3 has access logs, and you can store them in other s3 buckets. All API calls
  can be tracked via AWS CloudTrail. 
* You can also add tags to objects, and make policies for anything with a
  specific tag, e.g. marking documents "Classified". 

### Kinesis

This is a data streaming service. The competitor would be something like Apache
Kafka. If you see the phrase "real-time" on the test, it's probably a Kinesis
question. 

There are 4 different subofferings here. 

1. Kinesis streams - for low-latency streaming
1. Kinesis Analytics - for real-time analytics using SQL
1. Kinesis Firehose - more of a tool for getting data into a DB
1. Kinesis Video - for real-time video streaming. 

The basic use-case is roughly 

``` [Click Streams, IoT, Logs] > Kinesis Streams > Kinesis Analytics > DB ```

#### Kinesis Data Streams

Durability and availability here work a little bit differently. Data is retained
only for 24 hours. This gives you some ample time to replay a process if things
go wrong. But the AZ number is 3 and records can't be deleted. So for the access
window, data durability is still pretty high. Record size up to 1 MB. 

Streams again subdivides into two modes. 

> ** Provisioned Mode **: You choose the number of shards to provision. Each >
shard gets 1 MB/s in and 2 MB/s out. Price is per shard provisioned per hour. >
** On-demand Mode **: No need to provision shards, they are scaled automatically
> based on observed throughput peak during a 30 day window. Here you pay per >
stream per hour and per GB in/out. 

In addition, there are some following additional limits. 

* 1 MB/s per shard on input. If you go over this, you get
  "ProvisionedThroughputException" errors. 
* 2 MB/s per shard on output _across all consumers_. 
* 5 API calls per second per shard _across all consumers_. 
* Data is retained for 24 hours by default. This can be extended up to 365 days. 

I'm still not totally clear what a "shard" does for me in this context. It's
something I would probably need more familiarity with or to look at secondary
sources for. 

#### Kinesis Firehose

Kinesis is optimized mostly for throughput. It collects records from a wide
variety of producers and dumps it into a DB typically, or perhaps some other
http endpoint. Some light transformation work is possible via Lambda functions,
but not as much as other streaming offerings. And it is possible to backup all
records, or even just failed records, by writing the raw input out to S3 as
well.

Kinesis Firehose is fully managed, so it doesn't take any administrators on our
end to set up servers. It's not completely real-time, either. There's usually ~1
min latency between input and output at the minimum. In fact, the recommended is
~5 min. You pay based on the total throughput. 

The consumer side is a bit limited - data can only be deposited into Reshift,
S3, ElasticSearch, or Splunk DBs. It supports many data formats, though, along
with compression if the output is written to S3. And there aren't any
subdivisions based on partitioning here, the scaling is fully automatic. 

#### Comparisons

So which one do we want for what projects? Streams is more optimized for
applications and use. The draw is the lower latency and customizeable data
storage that allows for replaying streams and sending output to multiple
consumers. 

Firehose, by contrast, is more of an ingestion tool. There's more latency, and
fewer accepted consumers of the process, all of which are databases. There's
also no storage/replay option for Firehose, unless you also write out raw inputs
to some other DB for security. 

#### Kinesis Analytics

Can sit on the output of _either_ Data Streams or Firehose. Lets us query the
input streams as if they are database tables, and allows us to pull in reference
tables from S3 as well if we need. It's actually fuller-featured than even just
an analytics tool, though, as it also enables streaming ETL. 

It's an expensive tool, though, and you pay per resources consumed. But it does
scale automatically, support SQL and Flink to define the computation, and allow
lambda functions for pre-processing. 

The teacher shows an image of [Random Cut
Forest](https://assets.amazon.science/d2/71/046d0f3041bda0188021395b8f48/robust-random-cut-forest-based-anomaly-detection-on-streams.pdf),
which I've honestly never heard of but seems to be an anomaly detection algo
specifically designed to identify outliers in univariate time signals. But the
neat thing is that this can apparently been done through Kinesis Analytics on
live streams. So it's good to know that the methods we can apply here are fairly
robust. 

In the demo, Kinesis has a bunch of SQL templates for different analytics
offerings, and the teacher simply picks the anomaly detection template from
those offerings. The SQL dialect is... different. There are statements like
`CREATE OR REPLACE PUMP...`. 

#### Kinesis Video

Just a special stream service for video. The producers are typically some type
of IoT camera device, smartphone, etc. And this is piped into KVS. The
restriction here is one producer per stream. And many processes can stand in as
a consumer of video - sagemaker, ec2, Amazon's Rekognition Video service, etc. 

Data can be stored from up to 1 hour to 10 years. 

The teacher walks through an example architecture. Let's say you're building a
bodycam system to watch the cops. The bodycams pipe the feed into the KVS. A
[Fargate](https://aws.amazon.com/fargate/) serverless instance is set up to
consume the feed data and do the following:

1. Pipe checkpoints into a DB so if anything goes down, we know where to start
   again. 
1. Send decoded frames to an ML tool that's designed to identify use of force. 
1. Pipe `use_of_force = True/False` into a KDS with badge number, location, etc.
1. AWS lambda function to consume KDS feed and send out notifications when use
   of force is detected. 

### Glue

Glue is a tool that helps build and store a repository for metadata for all your
tables. It offers automated schema inferencing, schema versioning, and the use
of crawlers, which will presumably sit on your DB looking for changes and
integrate the data back into the catalog. 

Crawlers are pointed at some database (S3, Redshift, RDS) and go through the raw
files (JSON, Parquet, CSV, RS) to infer schemas and partitions. They can be
scheduled or run on demand, and they can be outfitted with certain credentials.

The thing to pay attention to is how your partitions are stored in S3. The
success of Glue really turns on this. It is effectively going to turn these
partitions into your schema/table/column info, so to get your metadata output
the way you want it, it must be stored in a specific way. But the appeal here is
setting up a store of raw files and getting out what you need to create a
database in a bush-button kind of way. 

Glue also has an ETL offering. It runs serverless, can be scheduled or set to
triggered on events, and is fully managed and fairly cheap. The transformation
offerings seem fairly basic - dropping fields, filters, joins, and maps. There's
also an ML-based offering to remove duplicate records. 

Glue is a DB Admin tool ultimately. And maybe a Mermaid project here to group
the various AWS offerings by their place in the DS project lifecycle here might
be a good exercise. 

### Athena

Athena appears to be a rough-and-ready DB/querying tool that lets you query data
directly from flat S3 files. It provides a querying tool that sits directly in
browser. You write a query and it writes out the results also to an S3 bucket. 

### Cleanup

To clean up everything Stephane just did, you delete the crawlers, you delete
everything in S3, you delete the delivery streams and Analytics objects, and you
delete the IAM roles. Any of our Athena work does not require the deletion of
any objects (other than the output files in S3), and our Kinesis streams could
in theory sit there because they don't cost money unless they're running. 

### AWS Data Stores

There's a couple offerings here - Redshift, RDS/Aurora, DynamoDB (NoSQL), S3,
OpenSearch, and ElastiCache. 

1. Redshift
    1. Heavy-hitter offering. Mostly for Data Warehousing and OLAP (online
       analytical processing)
    1. Data must be in S3 to start. And you typically load the S3 data into
       Redshift. However, you can use Redshift Spectrum as a tool that sits in
       between S3 and Redshift to dynamically query the data _from_ S3 and skip
       the loading. 
    1. Must be provisioned in advance. 
    1. Column-based storage. This makes it optimized to do heavy analytics on 
       large amounts of data in the cloud. 
1. RDS/Aurora
    1. This is the optimized-for-transactions offering (OLTP). Data is stored
       row-based. This makes it optimized to handle single transactions. 
    1. Must be provisioned in advance. 
1. DynamoDB
    1. NoSQL offering. 
    1. It's serverless, meaning no provisioning. You just need to specify the
       read/write capacity in advance. 
    1. Great for storing ML model output. 
1. S3 
    1. Serverless object storage. 
    1. Very much the backbone of anything AWS. 
1. OpenSearch (ElasticSearch)
    1. Great for indexing data and searching for observations. 
    1. Mostly used for clickstream analytics. 
1. ElastiCache
    1. Not really an ML thing. Mostly for data that has to be accessed real 
       fast for some online application. 

### AWS Data Pipelines

A different ETL offering. Usually writes out to S3, RDS, Dynamo, Redshift, or
EMR. It's a task orchestrator, think Airflow. The computation doesn't actually 
happen on the server running DP, it's merely calling out to an EC2 instance that
would do the computation. But this is how we typically manage task dependencies. 
Has some retry and failure notification processes. Data Sources can be on or
off premise. High availability with failover processes. 

So we could imagine some data sitting in RDS that needs to be moved into S3 to
be consumed by Sagemaker. In previous and current roles, I would simply have some
DB toolset in python or R that would run the query I need, and write out to S3. 
But if I wanted to do this in a purely-cloud-centric way, I could create a data
pipeline and ??feed it the same query??. Need to clarify that.

At any rate, I could use DP to manage the whole process, especially if the 
transformations were really intense and prone to failure. 

What's the difference between DP and Glue though? Glue actually runs code that 
you feeed it (Python, Scala, or Spark). However, you don't worry about the 
resources used - AWS takes care of all of that. DP, on the other hand, is more
about letting you control the resources provisioned. You don't feed it code, but
more tell it to spin up an EC2 instance that will run some github repo

### AWS Batch

Runs batch jobs off docker images. You don't actually provision the instances, 
this is totally services. You do, however, pay for what is provisioned. 

Like Glue, it's going to take care of all the resources for you. Unlike Glue, 
you specify the running environment via a docker container. In general, you're
using Glue for ETL, and you're using Batch for compute-intensive work that is 
_not_ ETL related. I'm not sure what that would actually entail in a cloud 
environment, Stephane points to running processes to "clean up" data sitting 
in an S3 bucket, but that still feels vaguely ETL-ish to me. 

### Database Migration Service

For, shockingly, database migration. We can think about DB migrations as
homogenous (Oracle on premise to Oracle in AWS) as well as Heterogenous (MS SQL
Server to Aurora), and there's support for both. 

The flow is a `Old DB > EC2 Running EMS > New DB`. There's no data transformation
here, so if you need some ETL work done you have to do it after the migration. 
But once the DB is in the cloud, you could use Glue to do the transforms. 

### AWS Step Functions

We use these to design workflows. They give us visualization of all the different
services that are being orchestrated, and offer more advanced error handling and
retry mechanisms then any single service as a standalone offering. 

Stephane shows an SF for an XGB model training job. It's all written in JSON, 
and defines at a very high level a process flow that goes 

``` Generate Data > Train > Save Model > Batch Transform to Score ```

It looks vaguely Airflow-ish, and the visualizations that are compiled off of 
the JSON are very nice. 

### Additional Topics

* AWS DataSync - System optimized to move on-prem storage to AWS. It works by 
employing a DataSync Agent, which is a virtual machine that sits on-prem and
connects to the storage. The agent then handles all the encryption and data 
validation. 
* MQTT - just a standard messaging protocol that handles IoT data. 





"""
