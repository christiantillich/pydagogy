#%% [markdown]
"""
# Objective

Discuss and demonstrate prompt chaining techniques in agentic AI systems, and
provide examples

# Overview

LLMs work best when given clear, simple, specific instructions. But often users
want to submit complex requests that are contingent on multiple factors. One way
to handle this complexity is to break down the request into smaller parts, each
handled by a specialized agent or prompt chain. A routing agent can analyze the
input and delegate sub-tasks to the appropriate specialized handlers. 

# Diagram

```mermaid
graph TD
    Input[User Request] --> Router[Router LLM]
    Router --> Decision{Classification}
    
    Decision -->|Category A| HandlerA[Specialized Handler A]
    Decision -->|Category B| HandlerB[Specialized Handler B]
    Decision -->|Category C| HandlerC[Specialized Handler C]
    Decision -->|Default| HandlerDefault[Default Handler]
    
    HandlerA --> OutputA[Response A]
    HandlerB --> OutputB[Response B]
    HandlerC --> OutputC[Response C]
    HandlerDefault --> OutputDefault[Default Response]
    
    style Input fill:#e1f5ff
    style Router fill:#fff9c4
    style Decision fill:#ffe0b2
    style HandlerA fill:#c8e6c9
    style HandlerB fill:#c8e6c9
    style HandlerC fill:#c8e6c9
    style HandlerDefault fill:#ffccbc
```

The key principle: a single routing LLM classifies the input, then conditionally executes one of several specialized handlers based on that classification.

# Example: Routing Requests to Specialized Handlers
"""

#%% [python]
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from IPython.display import Image, display

llm = ChatOpenAI(model="gpt-4", temperature=0)
def booking_handler(request: str) -> str:
    "Simulates the booking agent handling request."
    print("\n Delegating to booking agent...")
    return f"Booking Handler processed request: '{request}'. Result: Simulated booking action."

def info_handler(request: str) -> str:
    "Simulates the info agent handling request."
    print("\n Delegating to info handler...")
    return f"Info Handler processed request: '{request}'. Result: Simulated information retrieval."

def unclear_handler(request: str) -> str:
    "Handles requests that couldn't be delgated"
    print("\n Delegating to unclear handler...")
    return f"Coordinator could not delegate request: '{request}'."

coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system"
    ,""" 
        Analyze the user's request and determine which specialist handler process
        should handle it. 
        - If the request is related to booking, respond with 'booking'.
        - If the request is for travel information, respond with 'info'.
        - If the request is unclear or doesn't fit into either category, respond with 'unclear'.
        Only output one word: 'booking', 'info', or 'unclear'.
    """)
    ,("user", "{request}")
])

coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

branches = {
    "booker": RunnablePassthrough.assign(output = lambda x: booking_handler(x['request']['request']))
    ,"info": RunnablePassthrough.assign(output = lambda x: info_handler(x['request']['request']))
    ,"unclear": RunnablePassthrough.assign(output = lambda x: unclear_handler(x['request']['request']))
}

delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'booking', branches['booker']),
    (lambda x: x['decision'].strip() == 'info', branches['info']),
    branches['unclear']
)

coordinator_agent = {
    "request": RunnablePassthrough()
    ,"decision": coordinator_router_chain
} | delegation_branch | (lambda x: x['output'])

request_1 = "I would like to book a flight to London."
request_2 = "What is the capital of Italy?"
request_3 = "Tell me a joke about quantum physics."

coordinator_agent.invoke({"request": request_1})
coordinator_agent.invoke({"request": request_2})
coordinator_agent.invoke({"request": request_3})

#%% [markdown]
"""
Okay, so this is example was tricky for me at first. Start at
`coordinator_agent`. The whole object is a `RunnableSeries` made up of three
parts. The first part is a dictionary - when added to a `RunnableSeries`, a
dictionary actually gets converted to a runnable parallel object, where each
element runs in parallel and the results are collected into a dictionary. So the
first part has two keys, `decision` and `request`. The `decision` key is itself
the pipeline to prompt the LLM to classify the input. Here `request` just passes
through the input request unchanged. Now delegation branch gets both the LLM's
decision and the original request. 

We can actually view the whole pipeline like this:
"""

#%% [python]
coordinator_agent.get_graph().print_ascii()

#%% [python]
display(Image(coordinator_agent.get_graph().draw_mermaid_png()))

#%% [markdown]
"""
The next tricky part is `RunnableBranch`. It's like a switch. But you construct
it with tuples of (condition, runnable). So what you're seeing there is just us
checking the input key `decision` against the supported values, and then looking
up the corresponding runnable wrapper around the handler functions we defined at
the top.

`RunnablePassthrough.assign` is just a way to create a runnable that also
appends new values. Here the appended value is `output` and is the result of the
handler function. 
"""

#%% [python]
RunnablePassthrough.assign(
    output = (lambda x: "test")).invoke({"request": request_1}
)

#%% [markdown]
"""
So the whole `coordinator_agent` chain kindof works like this:

1. We submit the request as a dict. 
2. Passthrough adds that request directly, and evaluates that request with
   `coordinator_router_chain` to get the decision. 
3. The combined `request`/`decision` dict is passed to a runnable branch, which
   runs some functions to evaluate `decision` and pick the right handler.
4. The branch looks up the right handler function, which is itself wrapped in a 
   `RunnablePassthrough` that appends the handler's output as `output`.
5. Finally, we have a lambda at the end that just extracts the `output` key. 

But you could imagine us adding additional handling after this point. The
"booking" agent might itself be another LLM chain that starts inquiring about
flight dates, airline preferences, layover preferences, etc.  
"""