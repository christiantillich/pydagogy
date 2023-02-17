#%% [markdown]
"""
# Objective

Goal of [the paper]
(https://ml-and-vis.org/wp-content/uploads/2021/06/Cluster-clean-label_An-interactive-Machine-Learning-approach-for-labeling-high-dimensional-data_2020.pdf) 
appears to be evaluating a kind of automated workstream for 
unsupervised classification, and makes a nod to effectively utilizing the time
of SMEs like doctors or engineers. 

# Background

The paper seems to be combining a couple ideas that come from prior research

1. A cluster-then-label workflow described by Peikari et. al
1. A combination of t-SNE and PCA to get dimension reduction on high dimensional
   (image) data (Agis and Pozo)

# Methods

They call the method _cluster-clean-label_, and they use the MNIST data set
as the example to test. The "Cluster" part of the workflow itself is two segments -
dimensionality reduction and the actual clustering. The clustering pipeline goes

```
Raw Images > PCA > t-SNE > HDBSCAN
```

The cleaning pipeline then does the following workflow for each cluster. 

```
Auto-encoder > Identify Outliers > HitL to remove outliers and save cluster
```

The final part of the workflow then is labeling, which goes roughly 

```
Display Cleaned Cluster Instances > HitL to assign label
```

The advantage here is that users can see what is essentially a random sample
within the cluster now, and assign a label to the whole thing. This minimizes
the use of SME time. 

# Results

The overall result was that individual users of the tool achieved 99-100% 
accuracy in labeling digits 0-8, and 95-97% accuracy on labeling digit 9 (?!). 

# Conclusions

On a cursory read, I took this as a "let's throw every clustering method we 
have at the data and see what comes up". I think that initial impression is
incorrect, but overall, I'm still not convinced that some of these steps are
not redundant. t-SNE seems to perform very well on the MNIST data set all 
[on its own](https://lvdmaaten.github.io/tsne/examples/mnist_tsne.mov). Do we
really need PCA here too? 

Also, I suspect there's a world of difference between unsupervised labeling on 
MNIST and unsupervised labeling on a more-real data set. In MNIST, we know a 
priori there _should_ be nine clusters, and any user of this tool is going to 
be able to figure out the gambit pretty quick. In UCPs where the number of 
clusters isn't known _a priori_, I'm more skeptical that HDBSCAN is going to 
make the right choice and even more skeptical that we're going to nail the 
hyperparameter choice in a way that's generalizeable on the first try. In fact, 
there's a _lot_ of hyperparameter choices here that'll happen behind the 
scenes (why 86 PCA dimensions) that I don't feel comfortable about. 

Nevertheless, you can see the appeal if something like this did work. Imagine
a tool like this sitting alongside an analyst in Binder. I think there's a long
way to go to try and make that a reality, but the problem it's trying to solve
is obvious. Similarly I think we actually probably could have built this model
verbatim at Uptake and it would have been better than the classifications we
got from LDA. Maybe. At least, if it works as well on text data as it did 
images. 

"""