#%% [markdown]
"""
# Objective

To review [this paper](https://arxiv.org/pdf/2104.01136.pdf), which manages to 
merge convolutional neural architectures with transformers in a way that 
provides an attractive speed/accuracy tradeoff. 

# Background 

The authors start the discussion at Transformers, noting that the Transformer
model is a combination of MLP and multi-headed attention layers. The authors
then point to [ViT](https://arxiv.org/abs/2010.11929) and
[DeiT](https://arxiv.org/abs/2012.12877) as two recent attempts to incorporate
transformers into the image detection problem, which has previously been
dominated by convolutional architectures. The authors also note a bit later that
attempts at computer-vision using predominantly Transformer modules has lead to
the network imitating convolutions naturally. 

Because of this, the authors run an experiment - what happens when we cut off
the head of a ResNet-50 and frankenstein a transformer-only model on top of
that? And they show that with the same computational budget, the combination 
approach works better than either individually. This is the basis for the 
model the propose in Section 4. 

# Methods

The overall architecture is itself a patchwork of a number of different ideas.
In homage to their ResNet + DeiT hybrid, they start their model with 4 3x3
convolutional layers in series to compress the image. The image starts with 
3 channels at 224 pixels for length and width. The convolutions compress this
down and separate this out into many more chnanels, with the end result being
256 channels of images 14x14 pixels across. 

The model is then divided up into three stages, which have fundamentally the
same architecture, but differ in parameterization. In each stage, the feature
tensor is fead through a multi-headed attention block, normalized, and then 
fed through a perceptron, before being normalized again. The single stage is some
number of repetitions of this process. Between each stage is a shrinking 
attention block. 


# Questions

1. Am I getting this right? The structure here is all about taking a complex 
image and creating more channels that encode smaller blocks of features? So
3x224x224 gets broken down to 512 channels each 4x4? 



"""