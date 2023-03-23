#%% [markdown]
"""
# Objective

To review the [Document Understanding Transformer
(DonUT)](https://arxiv.org/pdf/2111.15664.pdf) model. 

# Background 

The goal is to build somethign that doesn't just read a document, but understands
it holistically. The current state of the art - and the trick that we employ 
here at Numerator nonetheless - is to first pass over the image with OCR, then
feed the OCR results into some downstream model, which then produces the desired
output, namely a structured json of all the information contained in the document.

The authors note that OCR has some problems - namely that it's a very expensive
preprocessing method. It's also generally pretty poor at transcribing pictographic
languages. It's clear here that much of the breakthrough of the Donut model is
driven by this need to transcribe Korean receipts. 

There's a second part of this paper, too, which is a data generation model that
the team also invented called SynthDoG. The authors use this to generate 
fake receipts as a pre-training step. 

# Methods

Donut seems to be a type of transformer model where the encoder step takes in
an image, and the decoder step takes in the encoded image along with some kind of
prompt. That prompt could ask the model to evaluate what type of document it is,
or to ask a question about the document, or to parse the information from the
document into a readable json. In this way, Donut is a system for document
understanding and comprehension, not merely OCR. 

Donut appears to build on the concept of a [Swin
Transformer](https://arxiv.org/pdf/2103.14030.pdf), which appears to be an
attempt to utilize the tricks of the transformer model applied to NLP problems,
and apply them to image problems. Specifically, Swins split the image into
little patches, and then blocks, where self-attention gets applied locally in
each little window. Furthermore, by changing the granularity of the window, the
Swin Transformer can organize the different feature maps hierarchically. 

Donut takes the image as input through this Swin Transformer, and outputs tokens
which are then encorporated alongside the user query in the decoder step. This
is essentially a process, then, for getting a LLM to incorporate the information
of an image as part of its output. The text tokens and visual tokens are all
incorporated to generate output tokens corresponding to specific words in a 
well-formatted JSON document. Through training, the model learns what words are
most likely to appear next in this json document. 

SynthDoG was a data generator tool they built, which I think is best described
as a type of meme generator tool, but I guess more boring. The precedent is 
[SynthTIGER](https://arxiv.org/pdf/2107.09313.pdf). They sample different
types of documents off ImageNet, then ripped numerous little text samples off
wikipedia and placed them over the blank image of the document. In the end, you
get something document-like, with full knowledge of the included text, complete
with auto-generated margins and layout. I believe they quoted making half a 
million documents in each of the target languages using this approach. 


"""