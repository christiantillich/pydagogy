#%% [markdown]
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

### Kinesis Data Streams

This is a data streaming service. The competitor would be something like 
Apache Kafka. If you see the phrase "real-time" on the test, it's probably a
Kinesis question. 

There are 4 different subofferings here. 

1. Kinesis streams - for low-latency streaming
1. Kinesis Analytics - for real-time analytics using SQL
1. Kinesis Firehose - more of a tool for getting data into a DB
1. Kinesis Video - for real-time video streaming. 

The basic use-case is roughly 

```
[Click Streams, IoT, Logs] > Kinesis Streams > Kinesis Analytics > DB
```

#### Kinesis Streams

Durability and availability here work a little bit differently. Data is retained
only for 24 hours. This gives you some ample time to replay a process if things
go wrong. But the AZ number is 3 and records can't be deleted. So for the access
window, data durability is still pretty high. Record size up to 1 MB. 

Streams again subdivides into two modes. 

> ** Provisioned Mode **: You choose the number of shards to provision. Each
> shard gets 1 MB/s in and 2 MB/s out. Price is per shard provisioned per hour. 
> ** On-demand Mode **: No need to provision shards, they are scaled automatically
> based on observed throughput peak during a 30 day window. Here you pay per
> stream per hour and per GB in/out. 

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
records, or even just failed records, by writing the raw input out to S3 as well.

Kinesis Firehose is fully managed, so it doesn't take any administrators on our
end to set up servers. It's not completely real-time, either. There's usually 
~1 min latency between input and output at the minimum. In fact, the recommended
is ~5 min. You pay based on the total throughput. 

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
fewer accepted consumers of the process, all of which are databases. There's also
no storage/replay option for Firehose, unless you also write out raw inputs to
some other DB for security. 

#### Kinesis Analytics
"""
