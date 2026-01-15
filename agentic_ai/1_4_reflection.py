#%% [markdown]
"""
# Objective

Discuss and demonstrate reflection techniques in agentic AI systems, and provide
examples of how agents can self-evaluate and improve their responses.

# Overview

LLMs can generate impressive responses, but they may sometimes produce errors or
suboptimal outputs. Reflection techniques allow agents to review their own
responses, identify potential issues, and refine their answers. This can be done
by having the agent generate a self-assessment or by using a secondary agent to
evaluate the initial response.

# Diagram

```mermaid
graph TD
    Input[User Request] --> Agent[Primary LLM Agent]
    Agent --> InitialResponse[Candidate Response]
    InitialResponse --> Agent
    Agent --> RefinedResponse[Refined Response]

    style Input fill:#e1f5ff
    style Agent fill:#fff9c4
    style InitialResponse fill:#ffe0b2

```

Here the LLM generates an initial response, then sends that response back to
itself with a prompt to reflect and improve upon it, resulting in a refined
answer.

# Example: 


"""

#%% [python]
import os 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Image, display

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from typing import TypedDict
# Instatiate LLM object
llm = ChatOpenAI(model="gpt-4", temperature=0.1)

# Save some string prompts for reuse. 
prompt = """
Your task is to create a Python function named `calculate_factorial`. 

This function should do the following:
1. Accept a single integer input `n`.
2. Calculate the factorial (n!)
3. Include a clear docstring explaining what the function does. 
4. Handle edge cases: The factorial of 0 is 1. 
5. Handle invalid input: Raise a ValueError if the input is a negative number. 

Please provide the python and only the python. Do not include explanations,
comments, or example usage. 
"""

reflection = """
You are a senior software engineer and an expert in Python. Your role is to
perform a meticulous code review. Critically evaluate the provided Python code
based on the original task requirements. 

Look for bugs, style issues, missing edge cases, and areas for improvement. If
the code is perfect and meets all requirements, please respond with the phrase
'CODE_IS_PERFECT'. Otherwise, provide a bulleted list of your critiques. Please
provide only the bulleted list - do not include a copy of the code, or anything
other than the critique and the bulleted list. 
"""

# Define prompt templates and chains
generation_prompt = ChatPromptTemplate.from_messages([
    ("human", "{task}")
])

critique_prompt = ChatPromptTemplate.from_messages([
    ("system", reflection),
    ("human", "Original Task:\n{task}\n\nGenerated Code:\n{code}")
])

refinement_prompt = ChatPromptTemplate.from_messages([
    ("human", "{task}"),
    ("assistant", "{code}"),
    ("human", "Critique:\n{critique}\n\nPlease refine the code using these critiques.")
])
generator = generation_prompt | llm | StrOutputParser()
critic = critique_prompt | llm | StrOutputParser()
refiner = refinement_prompt | llm | StrOutputParser()


def run_reflection_loop(task: str, max_iterations: int = 3) -> str:
    """Implement reflection using LangChain chains."""
    current_code = generator.invoke({"task": task})
    print(f"\n--- Iteration 1: Initial Generation ---\n\nCode:\n\n{current_code}\n")
    
    for i in range(1, max_iterations):
        # Critique the code
        critique = critic.invoke({"task": task, "code": current_code})
        print(f"Critique:\n{critique}\n")
        
        # Check if perfect
        if "CODE_IS_PERFECT" in critique:
            print("Code is perfect! Ending reflection.")
            break
        
        # Refine the code
        current_code = refiner.invoke({
            "task": task,
            "code": current_code,
            "critique": critique
        })
        print(f"--- Iteration {i+1}: Refined Code ---\n{current_code}\n\n")
    
    return current_code
        

#%% [python]
run_reflection_loop(prompt)

#%% [markdown]
"""
This is a modification of the book example, explicitly using LangChain chains to
implement the reflection loop. But the actual reflection logic isn't something
doable in LangChain directly. Reflection itself is something better handled with
regular python. 

There is nothing new in terms of LangChain usage here, just the same prompt
chains explored in earlier chapters. The novel part is that we embed these
chains in a loop that oscillates between code generation and
critique/refinement. 

There is a package called LangGraph. This _is_ capable of implementing
reflection as composition. 

We'll show this as an example below now. 
"""

#%% [python]
# Node functions
class ReflectionState(TypedDict):
    task: str
    current_code: str
    critique: str
    iteration: int
    max_iterations: int
    messages: list

def generate_code(state: ReflectionState) -> ReflectionState:
    """Generate or refine code based on current state."""
    iteration = state["iteration"]
    messages = state["messages"]
    
    if iteration == 0:
        print(f"\n--- Iteration {iteration + 1}: Initial Code Generation ---\n")
        messages = [HumanMessage(content=state["task"])]
    else:
        print(f"\n--- Iteration {iteration + 1}: Reflection and Improvement ---\n")
        messages.append(
            HumanMessage(content="Please refine the code using the critiques provided.")
        )
    
    response = llm.invoke(messages)
    current_code = response.content
    messages.append(response)
    
    print(f"Generated Code:\n\n{current_code}\n")
    
    return {
        **state,
        "current_code": current_code,
        "messages": messages,
    }

def reflect_on_code(state: ReflectionState) -> ReflectionState:
    """Critique the generated code."""
    critique_messages = [
        SystemMessage(content=reflection),
        HumanMessage(
            content=f"Original Task:\n{state['task']}\n\nGenerated Code:\n{state['current_code']}"
        )
    ]
    
    critique = llm.invoke(critique_messages).content
    print(f"Critique:\n\n{critique}\n")
    
    messages = state["messages"]
    messages.append(HumanMessage(content=f"Critique of previous code:\n{critique}"))
    
    return {
        **state,
        "critique": critique,
        "messages": messages,
        "iteration": state["iteration"] + 1,
    }

def should_continue(state: ReflectionState) -> str:
    """Decide whether to continue refining or end."""
    if "CODE_IS_PERFECT" in state["critique"]:
        print("\nCode is perfect! Ending reflection loop.")
        return "end"
    elif state["iteration"] >= state["max_iterations"]:
        print(f"\nReached max iterations ({state['max_iterations']}). Ending loop.")
        return "end"
    else:
        return "continue"

def build_reflection_graph():
    """Construct the reflection workflow graph."""
    workflow = StateGraph(ReflectionState)
    
    # Add nodes
    workflow.add_node("generate", generate_code)
    workflow.add_node("reflect", reflect_on_code)
    
    # Add edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "continue": "generate",
            "end": END,
        }
    )
    return workflow.compile()

#%% [python]
graph = build_reflection_graph()
final_state = graph.invoke({
    "task": prompt,
    "current_code": "",
    "critique": "",
    "iteration": 0,
    "max_iterations": 3,
    "messages": [],
})
final_state

#%% [python]
display(Image(graph.get_graph().draw_mermaid_png()))


#%% [markdown]
"""
So this is much more complex, but there's a good reason. Using LangGraph, we can
actually manage a stated process flow. This process tracks previous iterations,
and updates the conversation based on those previous iterations. This is
something that LangChain alone cannot do, as it lacks the ability to maintain
state across repeated passes. 

But LangGraph allows us to manage stated, complext workflows while still being
composable. The cost for that, though, is a more manual attitude w.r.t. managing
the conversation history. 
"""
