#%% [markdown]
"""
# Objective

Discuss and demonstrate parallelization techniques in agentic AI systems, and
provide examples. 

# Overview

LLM requests often require some overhead. For example, a user might need to
review 100 independent documents and evaluate each for fraud markers. In such a
case, we do not need to process the documents sequentially; instead, we can
process them in parallel, significantly speeding up the overall task. 

# Diagram

```mermaid
graph TD
    Input[User Request: Process Multiple Items] --> Prep[Prepare Tasks]
    Prep --> Split[Split into Independent Tasks]
    
    Split --> Task1[Task 1<br/>Process Doc A]
    Split --> Task2[Task 2<br/>Process Doc B]
    Split --> Task3[Task 3<br/>Process Doc C]
    Split --> TaskN[Task N<br/>Process Doc N]
    
    Task1 --> LLM1[LLM Call 1]
    Task2 --> LLM2[LLM Call 2]
    Task3 --> LLM3[LLM Call 3]
    TaskN --> LLMN[LLM Call N]
    
    LLM1 --> Result1[Result 1]
    LLM2 --> Result2[Result 2]
    LLM3 --> Result3[Result 3]
    LLMN --> ResultN[Result N]
    
    Result1 --> Aggregate[Aggregate Results]
    Result2 --> Aggregate
    Result3 --> Aggregate
    ResultN --> Aggregate
    
    Aggregate --> Final[Final Output]
    
    style Input fill:#e1f5ff
    style Prep fill:#fff9c4
    style Split fill:#ffe0b2
    style Task1 fill:#c8e6c9
    style Task2 fill:#c8e6c9
    style Task3 fill:#c8e6c9
    style TaskN fill:#c8e6c9
    style LLM1 fill:#b3e5fc
    style LLM2 fill:#b3e5fc
    style LLM3 fill:#b3e5fc
    style LLMN fill:#b3e5fc
    style Aggregate fill:#fff9c4
    style Final fill:#c5e1a5
    
    classDef parallel fill:#c8e6c9,stroke:#4caf50,stroke-width:3px
    class Task1,Task2,Task3,TaskN,LLM1,LLM2,LLM3,LLMN,Result1,Result2,Result3,ResultN parallel
```

The key principle: independent tasks are processed concurrently by multiple LLM calls, then results are aggregated. This dramatically reduces total processing time compared to sequential execution.

# Example: Parallel Document Analysis
"""
#%% [python]
import os
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough 

from IPython.display import Image, display

llm = ChatOpenAI(model="gpt-4", temperature=0)

summarize_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize the following topic concisely:")
        ,("user", "{topic}")
    ])
    | llm 
    | StrOutputParser()
)

questions_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Generate three insightful questions about the following topic:")
        ,("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

terms_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identify 5-10 key terms from the following topic, separated by commas:")
        ,("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

map_chain = RunnableParallel({
    "summary": summarize_chain,
    "questions": questions_chain,
    "key_terms": terms_chain,
    "topic": RunnablePassthrough()
})

main_template = """
Based on the following information:
* Summary: {summary}
* Related Questions: {questions}
* Key Terms: {key_terms}

Synthesize a comprehensive answer. 
"""

synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", main_template)
    ,("user", "Original topic: {topic}")
])

full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()
display(Image(full_parallel_chain.get_graph().draw_mermaid_png()))

#%% [python]
climate_topic = "The impact of climate change on global agriculture."
_ = print(asyncio.run(full_parallel_chain.ainvoke(climate_topic)))


#%% [markdown]
"""
So, what I'd like to do in this section is kindof pick apart the chain and show
how these elements works. _However_, `asyncio` actually really struggles with
running multiple `run` calls in a single session. So for this one doc, I'm just
going to explain the pieces, and you can run the whole thing in a fresh session
to see how it works. 

map_chain is a `RunnableParallel` object. It defines a RunnableParallel with
three branches and a passthrough element. Each branch is itself a prompt chain
doing different independent tasks using the same topic. So the output here is
going to be a dictionary with four keys: `summary`, `questions`, `key_terms`,
and `topic`.

Again, I can't rerun just a part of the chain, this is the actual output of just
`map_chain.invoke(climate_topic)`:
"""

#%% [python]
{'summary': """
    Climate change significantly impacts global agriculture by altering weather
    patterns, increasing frequency and intensity of extreme weather events, and
    raising global temperatures. These changes can lead to reduced crop yields
    and livestock productivity, shifts in planting and harvesting times, and
    increased pests and diseases. Additionally, climate change can exacerbate
    water scarcity, soil degradation, and biodiversity loss, further threatening
    food security. However, it also opens possibilities for agricultural
    adaptation and mitigation strategies, such as climate-smart agriculture,
    precision farming, and sustainable land management.
    """
 ,'questions': """
    1. How is climate change affecting the productivity and sustainability of
       global agriculture?
    2. What are the potential long-term impacts of climate change on food
       security worldwide?
    3. How can agricultural practices be adapted or modified to mitigate the
       effects of climate change?
    """
 ,'key_terms': """
    Climate change, global agriculture, impact, greenhouse gases,
    global warming, crop yield, adaptation strategies, soil degradation, water
    scarcity, food security.
    """
 ,'topic': 'The impact of climate change on global agriculture.'
}

#%% [markdown]
"""
So when we feed this dictionary into `synthesis_prompt`, the dictionary items
plug neatly into the four expected input values. In particular, note the
difference between `summary` and the final answer. Both are really just valid
responses to the prompt. But by holding a brief summary, insightful questions,
and key terms in the model's context independently, and then asking it to
synthesize a final answer, we get a much more comprehensive and higher-quality final
response.
"""
