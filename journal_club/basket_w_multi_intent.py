#%% [markdown]
"""
# Objective

The goal was to [apply intention](https://arxiv.org/pdf/2010.11419.pdf) to 
modern NN-based attempts at the Basket Recommendation problem. This comes off
a conversation with Trent Baer and so I recommended this paper to the ML team
to review. 

# Background

I want to tackle background 2 ways here. There's currently a specific business
need for a report in Insights. The idea would be to choose a majorcat and get
a kind of decision tree back. The terminal points are groups of items that 
tend to stand as substitutions for each other. The branch points represent a
kind of decision or preference that needed to be made. In Trent's example, 
the majorcat "soda" split off into a decision about bottles or cans, then cola
or citrus. The terminal groups, left to right, were `[2L Coke, 2L Diet Coke]`, 
`[12ct Sprite, 12ct MtDew]`, and `[12ct Coke, 12ct Diet Coke]`. This seems like
a BR problem, where a report might sit in front of a model prompting the model
with a majorcat and a list of people groups, and the model might do some 
prediction and generate a list of recommended items. 

This prompted me to start up a search for the latest in recommender engines, and
that's how I came upon the paper. Having just learned about attention, I thought
it might be useful to to explore this approach to BR problems. 

Overall, the argument the paper makes is compelling - stating essentially that
attention in BR problems seems like a necessary component, because different 
items can be correlated differently in different contexts. Attention handles
the context. A basket might be fulfilling latent desires like "house-cleaning", 
"breakfast", and "burger". We want a model to identify to pick up on 
complementary and subsitutional relationships - e.g. cheese slices complements
burger patties, but sometimes we trade out american for cheddar. And we don't
want to recommend dish detergent at all if the need is "burger". There are other
ways of exploring these relationships, but Attention-based methods feel like a
natural fit and the authors note that it had not yet been explored. 

The authors use a number of different models as reference. 

* BPR-MF - Bayesian Personalized Ranking.
* Triple2vec - A way of embedding items that are typically found in the same basket. 
* DREAM - A RNN that adds in user-embeddings to item-embeddings to get a more
    holistic picture of the basket and needs. 
* R-GCN - A Graph Convolutional Network approach to the BR problem. 
* GC-MC - Graph Convolutional Matrix Completion
* NGCF - Neural Graph Collaborative Filtering

# Method

The general structure of the model goes as follows. Inputs are user-embeddings,
basket embeddings, and item embeddings. These inputs then go through a traditional
attention layer, but the output splits into different streams. We recover 
a component of the attention specifically for the user and items, and these are
routed off to aggregators that take all the embeddings as input, mix with 
attention, and then output embeddings for the user and item directly. 

It's not clear to me here why the basket embedding just "passes through". Intention
mixes with user embeddings and item embeddings to produce "x-guided" embeddings
for each. It's not totally clear to me why this doesn't also happen with the
baskets. 


# Results

They look convincing to me. There's two sets of grocery data, and both appear
to be significantly robust at the surface level. Panel sizes also seem to be
robust, at 100k and 200k panelists each and millions of transactions. Success
metrics used are 

* Recall@K - for K recommended items, what percentage of items were in the 
    recommended list. 
* HR@K - Hit Ratio. For K recommended items, what percentage of users did a
    purchase item exist in the recommended list. 
* NDCG@K - Normalized Discounted Cumulative Gain. A measure of how highly relevant
    your ranked suggestions are, with a reward/penalty assigned based on the
    actual utility vs the position in the ranking system. 

They make sense. I'm not in a position to judge if we're leaving things out, here, 
but if I was building a recommender it seems intuitive that I would want these
at least. 

MITGNN is the clear winner in every test. Although I'm kindof surprised to see
the wildly different baselines between Instacart and Walmart data sets. I'm not
sure why this would be, and it's especially surprising to see Recall and NDCG
_an order of magnitude lower_ for Walmart, where I would assume the number of
purchaseable items to be smaller. Perhaps this is because Instacart is mostly
used for food, whereas Walmart shopping can produce significantly more basket
contexts? I would have liked some kind of comment here. 

# Questions

* I think I'm clear on item-embeddings, but what is a user embedding in this 
    context?
* I think I kindof understand why we get user-guided and item-guided embeddings as
    the product of attention. Why would we not also get basket-guided imbeddings. 


"""