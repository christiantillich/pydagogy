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
"""
