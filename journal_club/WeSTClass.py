#%% [markdown]
"""
# Objective

Review the paper on [WeSTClass](https://arxiv.org/pdf/1809.01478.pdf). 

# Background

We all know supervised and unsupervised learning. This paper continues knowledge
in a space called "Weakly Supervised Learning", applying the concepts to a text
classification problem. The idea behind weakly supervised learning is, roughly,
to use some ML process or crude way of _assigning_ labels, and then train a Deep
Learner to estimate the very-noisy labels. The thought is that the Deep Learner
will actually eat away at some of the error and produce more accurate results
than the Shallow Learner used to generate the labels. University of California
first showed this concept when trying to construct [3D images of faces from
photos alone](https://arxiv.org/pdf/1612.04904.pdf). In a way, this points to a
recurring interest in synthetic data generation. 

At the highest level, weakly supervised learning problems need a Shallow Learner
to create the labels and a Deep Learner to learn them. The shallow learner here
is something called a ["Skip-Gram" model](
https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-skip-gram.html
). The core problem that a SGM is trying to solve is "what words occur next to
other words in sentances". So in a sentence like

> The quick brown fox jumps over the lazy dog. 

The prediction problems is `x="fox"` and `y=["quick","brown","jumps","over"]`.
The input is a word and the output is the context. CBOW models, by contrast, go
the other way and try to predict the word in the middle. 

The Deep Learner varies, the experimenters play with CNNs and HANs, and I'm
still trying to wrap my head around what that means from the perspective of
architecture. But they note that this is really an umbrella method that could
support a wide variety of Deep Learners. 

# Methods

This paper is actually more like 3 different papers, because the authors concern
themselves with three different methods of generating labels from the small
corpus of real documents. 

1. Very-broad label names that apply generally to the whole class - e.g.
   Politics, Sports, etc. 
2. Broad class-related keywords that apply generally the whole class - e.g.
   `Class1 = {'democracy','republic','religion','liberal'}. 
3. Sets where each document has a label from (1), but assigned individually. 

It's not clear to me how (3) is different from (1), and that's a good topic for
questions later, but I trust that these different labeling schemes are all
different, and that the author is pointing out implementation difficulties that
I don't yet see. 

I'm also kindof fuzzy on how the SGM gets us to the vMD. I know that vMD is a
high-dimensional unit-sphere where each axis is one of the learned topics. But
once we have it, we can simply sample from that distribution at random to 
generate a new document. There seems to be some additional math for describing
how to generate the words _within_ that document - e.g. we need to sample words
from the class-specific distribution `f` but also some backgound distribution
that simply represents the likelihood of drawing a common word. In full english, 
the algorithm seems to go something like

> Take as input the unit sphere, the average length of documents, and the number
> of documents you want. For each class of interest, and for as many times as
> you want documents, sample the significant terms from our label-significant
> terms mapping, and then for each word in the document, sample from the mixing
> distribution that blends topic-specific words with background words. Congrats
> on your new document. 

The idea is to run this algorithm on blast, generate millions of documents and
then hit it with the NN-du-jour. 

# Questions

1. What is the role of the Skip-Gram model here? We're using it to map from 
words in a document to either the label or the class-related key-words?
1. How do we use the SGM to get to the vMD?

"""