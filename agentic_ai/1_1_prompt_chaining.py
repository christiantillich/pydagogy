#%% [markdown]
"""
# Objective

Discuss and demonstrate prompt chaining techniques in agentic AI systems, and
provide examples

# Overview

LLMs work best when given clear, simple, specific instructions. Large, complex
paragraphs are prone to misinterpretation and can lead to suboptimal results.
Prompt chaining is a pattern whereby the developer breaks down complex tasks
into smaller, more manageable sub-tasks, each with its own prompt. When prompt
chaining, the output of one prompt can be used as the input for the next prompt. 

# Diagram

```mermaid
graph LR
    Input[Initial Input] --> P1[Prompt 1]
    
    subgraph Chain1["Chain 1: First Sub-task"]
        P1[Prompt 1]
        LLM1[LLM Call 1]
        O1[Output 1]
        P1 --> LLM1
        LLM1 --> O1
    end
    
    O1 --> P2[Prompt 2]
    
    subgraph Chain2["Chain 2: Second Sub-task"]
        P2[Prompt 2]
        LLM2[LLM Call 2]
        O2[Output 2]
        P2 --> LLM2
        LLM2 --> O2
    end
    
    O2 --> P3[Prompt N]
    
    subgraph ChainN["Chain N: Final Sub-task"]
        P3[...]
        P3 --> LLM3[...]
    end
    
    LLM3 --> Final[Final Output]
    
    style Input fill:#e1f5ff
    style Final fill:#c8e6c9
    style LLM1 fill:#fff9c4
    style LLM2 fill:#fff9c4
    style LLM3 fill:#fff9c4
    style O1 fill:#ffe0b2
    style O2 fill:#ffe0b2
```

The key principle: each step's output becomes the next step's input, forming a
sequential chain of focused sub-tasks.

# Examples

Below, we show an example of a prompt chain designed to extract the technical
specifications from text and transform it into structured JSON data. 
"""

#%% [python]
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt_extract = ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text: \n\n{text}"
)
prompt_transform = ChatPromptTemplate.from_template(
    "Transform the following specifications into a JSON template with only " + 
    "the keys 'CPU', 'RAM', and 'Storage': \n\n{specs}"
)

extraction_chain = prompt_extract | llm | StrOutputParser()

full_chain = (
    {"specs": extraction_chain}
    | prompt_transform
    | llm
    | JsonOutputParser()
)

spec_1 = """
The new SuperFast Laptop comes with a powerful Intel i7 processor, 16GB of RAM
, and a spacious 512GB SSD for all your storage needs.
"""

spec_2 = """
This desktop computer features a kitted-out AMD Ryzen 5 CPU, NVIDIA RTX 4090
GPU, 128GB of RAM, and 2TB of SSD storage. 
"""

#%% [python]
full_chain.invoke({"text": spec_1})

#%% [python]
full_chain.invoke({"text": spec_2})

#%% [markdown]
"""
Note how the prompt chain returns a clean, formatted JSON object with only the
specified keys. What is going on here?

1. `prompt_extract` is a template. `.from_template` is going to create a whole
   chat from a single string assumed to be the human, but will evaluate `text`
   later when `.invoke` is called.
"""

#%% [python]
prompt_extract.invoke({"text": spec_1})

#%% [python]
(prompt_extract | llm).invoke({"text": spec_1})

#%% [python]
(prompt_extract | llm | StrOutputParser()).invoke({"text": spec_1})


#%% [markdown]
"""
2. `| llm` adds the llm evaluation to the chain. The whole chain is not
   executed, this is like sklearn. We're just specifying that we pass filled-in
   template to the LLM for evaluation. 
3. `StrOutputParser` is a simple output parser essentially just returns
   `.content` of the message object returned by the LLM and returns it as
   string.
4. we set extraction_chain to be the input to the transformation chain. 
5. Then we add a new prompt, another LLM evaluation, and finally a json parser. 

Note, `JsonOutputParser` does not _turn_ the output into json. It assumes the
output is formatted like a json, and wraps it in `json.loads`. If the llm is not
returning valid json already, you get an error. 
"""




