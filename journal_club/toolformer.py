#%% [markdown]
"""
# Objective

Review [this paper](https://arxiv.org/pdf/2302.04761.pdf) describing how to 
teach Transformer models how to use tools. Scary shit. 

# Background

Just kindof feels like the next natural evolution of where LLMs are going, but 
the author notes that [previous](https://aclanthology.org/2022.acl-long.579.pdf)
[approaches](https://arxiv.org/pdf/2211.10435.pdf) required too much human 
intervention or were too limited in scope. The objective of the paper is to 
get Transformer models to adopt tools in 

1. A self-supervised manner that requires no human markup
2. A way that is fully general, and allows the model to decide when to use which
   tool. 

Those seem like the same objectives to me honestly, but that's how the author
expressed them. 

# Methods

Overall, the authors seem to take the following approach to the model
construction.

1. Have a Language Model annotate a giant dataset with potential API calls. 
2. Use a self-supervised loss function to determine which API calls help 
   the model predict future tokens. 
3. Finetune the Launguage model while including the results from the API calls
   so that it can learn what is considered helpful. 

(1) here is tricky, but I think what they do is they first seed the LLM with 
the following prompt. Think of this like a function

```
def prompt(x, api_type):
    "Your task is to add calls to a " + api_type + " to a
    piece of text. The questions should help you get
    information required to complete the text. You can call the
    API by writing '[QA(question)]' where 'question' is the
    question you want to ask. Here are some examples of API
    calls:

    Input: Joe Biden was born in Scranton, Pennsylvania.
    Output: Joe Biden was born in [QA('Where was Joe
    Biden born?')] Scranton, [QA('In which state is
    Scranton?')] Pennsylvania.

    Input: Coca-Cola, or Coke, is a carbonated soft drink
    manufactured by the Coca-Cola Company.
    Output: Coca-Cola, or [QA('What other name is
    Coca-Cola known by?')] Coke, is a carbonated soft drink
    manufactured by [QA('Who manufactures Coca-Cola?')]
    the Coca-Cola Company.

    Input:" +  x + "
    Output:"
```

The first pass over the data will use the LM to predict _what the call to the 
API should be_. That's clever. Thus, we use the Oracle-in-Delphi hack to 
determine what the prompts to the API even should be. As they said, this workflow
lets you generate a ton of prompts. 

So now you've got a crap ton of prompts, so you execute them all. Simple enough.

Then there is a filtering step, where they try to determine the cross entropy 
loss over the tokens `x` while appending the tokens from the API call. API calls
are kept if they improve the marginal loss by some threshold $\tau$. 

The kept API calls are then merged into the actual corpus $C$ to produce a
modified corpus $C^*$. And they're merged in a very specific way, using the
format of `api_name(call) -> response`. The LM is then finetuned on the new
corpus $C^*$. 

Inference mode would then proceed as normal until the LM predicts a `->` should
be next. At this point, normal inference breaks and the specified API call is
made with the given prompt. That response is then inserted into the model output, 
along with a flag to note that the API call was made. 

Presumably, there's some pretty citation formatting code to pretty up the whole
thing before the LM user sees it. But this is how you do the citatation thing 
that Bing was doing. Very clever. 

I'm _inferring_ here that this is all done as part of a single training epoch,
and is not simply done as a one-time shot. I could be mistaken. 

# Questions

1. There are two "desiderata", but they seem like fundamentally the same thing. 
   Is anyone seeing a difference here that I just am not?
2. I'm guessing that pages 2-4 kindof describe what a single _training epoch_
   looks like. Is that the impression that other people got here too?

"""