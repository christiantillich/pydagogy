#%% [markdown]
"""
# Objective

Discuss and demonstrate the tool use pattern in agentic AI systems, and provide
examples of how agents can leverage external tools to enhance their
capabilities.

# Overview

If we strictly asked an LLM to look up data such as current stock prices or
weather, it would likely generate a plausible-sounding but ultimately incorrect
response. This is because LLMs are trained on static datasets and do not have
access to real-time information. To overcome this limitation, we can integrate
external tools or APIs that provide up-to-date data. The agent can then decide
when to invoke these tools based on the user's request.

# Diagram 

```mermaid
graph TD
    Input[User Request] --> Agent[Primary LLM Agent]
    Agent --> Tool[External Tool/API]
    Tool --> ToolResponse[Tool Response]
    Agent --> FinalResponse[Final Response]

    style Input fill:#e1f5ff
    style Agent fill:#fff9c4
    style Tool fill:#c8e6c9
    style ToolResponse fill:#ffe0b2
```

Here the LLM agent receives a user request, determines that it needs to use an
external tool to fulfill the request, invokes that tool, and then incorporates
the tool's response into its final answer.

# Example:
"""

#%% [python]
import os
from crewai import Agent, Task, Crew
from crewai.tools import tool
import logging

# To turn off crewai's tracing
os.environ["CREWAI_TRACING_ENABLED"] = "false"

@tool("Stock Price Lookup Tool")
def get_stock_price(ticker: str) -> float:
    """
    Fetches the latest simulated stock price for a given stock ticker symbol.
    Returns the price as a float. Raises a ValueError if the ticker is not
    found.
    """
    logging.info(f"Tool Call: get_stock_price for ticker '{ticker}'")
    simulated_prices = {
        "AAPL": 178.15,
        "GOOGL": 1750.30,
        "MSFT": 425.50,
    }
    price = simulated_prices.get(ticker.upper())
    if price is not None:
        return price
    else:
        # Raising a specific error is better than returning a string.
        # The agent is equipped to handle exceptions and can decide on
        # the next action.
        raise ValueError(
            f"Simulated price for ticker '{ticker.upper()}' not found."
        )

financial_analyst_agent = Agent(
    role='Senior Financial Analyst'
    ,goal='Analyze stock data using provided tools and report key prices.'
    ,backstory="""
        You are an experienced financial analyst adept at using
        data sources to find stock information. You provide clear, direct
        answers.
    """
    ,verbose=True
    ,tools=[get_stock_price]
    ,allow_delegation=False
)

analyze_aapl_task = Task(
    description = """
        What is the current simulated stock price for Apple (ticker: AAPL)? Use
        the 'Stock Price Lookup Tool' to find it. If the ticker is not found,
        you must report that you were unable to retrieve the price.
    """
    ,
    expected_output="""
        A single, clear sentence stating the simulated stock price for AAPL. For
        example: 'The simulated stock price for AAPL is $178.15.' If the price
        cannot be found, state that clearly.
    """,
    agent=financial_analyst_agent,
)

financial_crew = Crew(
    agents=[financial_analyst_agent],
    tasks=[analyze_aapl_task],
    verbose=False # Set to False for less detailed logs in production
)

thing = financial_crew.kickoff()
thing

#%% [markdown]
"""
That's the tool in a nutshell. At first glance, it kindof appears like CrewAI
lets you define a pool of agents and assign it tasks. You'd think based on this
example that you could set this up in a dynamic way - establish a set of agents
like a server waiting to respond to client requests, than process those
requests. 

However, in actuality, it really is an orchestration tool at heart - something
more like airflow. So the agents and tasks are assigned statically, then under
the hood `Crew` determines the best way to solve the problem. 

Because of this, we can actually develop some code to view the orchestration. 
"""


#%% [markdown]
"""
# Visualizing the Workflow

You can visualize the crew's workflow as a Mermaid diagram. Below I show some
functions for how to do this. 
"""

#%% [python]

def visualize_crew_workflow(crew: Crew) -> str:
    """
    Generate a Mermaid diagram representing the crew's workflow.
    
    Args:
        crew: A CrewAI Crew object with agents and tasks
        
    Returns:
        A string containing Mermaid markdown syntax for the workflow diagram
    """
    lines = ["```mermaid", "graph TD"]
    
    # Create agent nodes
    agent_ids = {}
    for i, agent in enumerate(crew.agents):
        agent_id = f"Agent{i}"
        agent_ids[agent] = agent_id
        role = agent.role.replace('"', "'")
        lines.append(f'    {agent_id}["{role}"]')
        lines.append(f'    style {agent_id} fill:#fff9c4,stroke:#f9a825')
    
    # Create task nodes and track dependencies
    task_ids = {}
    for i, task in enumerate(crew.tasks):
        task_id = f"Task{i}"
        task_ids[task] = task_id
        
        # Truncate description for readability
        desc = task.description.strip().replace('\n', ' ')[:50]
        if len(task.description.strip()) > 50:
            desc += "..."
        desc = desc.replace('"', "'")
        
        lines.append(f'    {task_id}["{desc}"]')
        lines.append(f'    style {task_id} fill:#e1f5ff,stroke:#0277bd')
        
        # Connect agent to task
        if task.agent and task.agent in agent_ids:
            agent_id = agent_ids[task.agent]
            lines.append(f'    {agent_id} --> {task_id}')
        
        # Add task context dependencies (task dependencies)
        if hasattr(task, 'context') and task.context:
            # Check if context is actually a list/iterable (not _NotSpecified)
            try:
                if isinstance(task.context, list):
                    for ctx_task in task.context:
                        if ctx_task in task_ids:
                            lines.append(f'    {task_ids[ctx_task]} --> {task_id}')
            except TypeError:
                pass  # context is not iterable
    
    # Add sequential flow if applicable
    if hasattr(crew, 'process') and str(crew.process) == 'Process.sequential':
        for i in range(len(crew.tasks) - 1):
            task_id = task_ids[crew.tasks[i]]
            next_task_id = task_ids[crew.tasks[i + 1]]
            lines.append(f'    {task_id} -.Sequential.-> {next_task_id}')
    
    lines.append("```")
    return '\n'.join(lines)

def display_crew_workflow(crew: Crew):
    """
    Display a Mermaid diagram of the crew's workflow in Jupyter.
    
    Args:
        crew: A CrewAI Crew object with agents and tasks
        
    Returns:
        IPython.display.HTML object that renders the Mermaid diagram
    """
    from IPython.display import HTML
    import uuid
    
    # Get the mermaid code without the markdown code fence
    diagram_text = visualize_crew_workflow(crew)
    # Remove ```mermaid and ``` wrappers
    mermaid_code = '\n'.join(diagram_text.split('\n')[1:-1])
    
    # Generate unique ID for this diagram
    diagram_id = f"mermaid-{uuid.uuid4().hex[:8]}"
    
    html = f"""
    <div id="{diagram_id}"></div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: false }});
        const graphDefinition = `{mermaid_code}`;
        const {{ svg }} = await mermaid.render('graph-{diagram_id}', graphDefinition);
        document.getElementById('{diagram_id}').innerHTML = svg;
    </script>"""
    return HTML(html)

#%% [python]
# Display as rendered diagram in Jupyter
display_crew_workflow(financial_crew)

#%% [markdown]
"""
Visually this workflow isn't much impressive. So let's generate a more complex
workflow using tools so that we can see how CrewAI would delegate the tasks. 
"""

#%% [markdown]
"""
# Complex Example: Market News-Driven Trading Analysis

In this example, we'll build a more sophisticated crew that:
- Monitors stock prices and alerts
- Analyzes news sentiment
- Assesses volatility and risk
- Synthesizes all data into trading recommendations

This demonstrates multiple agents working in parallel with task dependencies.
"""

#%% [python]
from crewai import Process
import random

# Define additional tools for the complex workflow

@tool("News Headlines Fetcher")
def fetch_news_headlines(ticker: str) -> str:
    """
    Fetches simulated recent news headlines for a given stock ticker.
    Returns headlines as a newline-separated string.
    """
    logging.info(f"Tool Call: fetch_news_headlines for ticker '{ticker}'")
    
    news_database = {
        "AAPL": [
            "Apple unveils groundbreaking AI features in latest update",
            "iPhone sales exceed analyst expectations",
            "Apple expands services revenue to record highs",
            "Minor supply chain disruptions reported for Apple",
            "Apple announces new sustainability initiatives"
        ],
        "GOOGL": [
            "Google's cloud division shows strong growth",
            "Regulatory scrutiny increases for Google search",
            "Google AI breakthrough in language models",
            "Advertising revenue slightly below forecasts",
            "Google invests $2B in renewable energy"
        ],
        "MSFT": [
            "Microsoft Azure gains market share in cloud computing",
            "Strong enterprise adoption of Microsoft 365",
            "Microsoft gaming division reports record quarter",
            "New AI partnerships announced by Microsoft",
            "Microsoft completes major acquisition successfully"
        ]
    }
    
    headlines = news_database.get(ticker.upper(), [
        f"General market conditions favor {ticker}",
        f"{ticker} maintains steady performance",
        f"Industry trends impact {ticker} outlook"
    ])
    
    # Return 3-4 random headlines as newline-separated string
    selected = random.sample(headlines, min(4, len(headlines)))
    return '\n'.join(selected)

@tool("Sentiment Analyzer")
def analyze_sentiment(headlines: str) -> dict:
    """
    Analyzes sentiment of news headlines and returns a sentiment score.
    Takes headlines as a newline-separated string.
    Returns dict with positive/negative/neutral counts and overall score.
    """
    # Convert string to list if needed
    if isinstance(headlines, str):
        headlines = [h.strip() for h in headlines.split('\n') if h.strip()]
    
    logging.info(f"Tool Call: analyze_sentiment for {len(headlines)} headlines")
    
    # Simulate sentiment analysis
    positive_keywords = ["breakthrough", "exceeds", "strong", "record", "gains", 
                        "unveils", "growth", "success", "completes"]
    negative_keywords = ["scrutiny", "disruptions", "below", "concerns", "decline"]
    
    sentiment_scores = []
    for headline in headlines:
        headline_lower = headline.lower()
        if any(word in headline_lower for word in positive_keywords):
            sentiment_scores.append(1)
        elif any(word in headline_lower for word in negative_keywords):
            sentiment_scores.append(-1)
        else:
            sentiment_scores.append(0)
    
    positive = sentiment_scores.count(1)
    negative = sentiment_scores.count(-1)
    neutral = sentiment_scores.count(0)
    overall_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "overall_score": round(overall_score, 2),
        "interpretation": "positive" if overall_score > 0.2 else "negative" if overall_score < -0.2 else "neutral"
    }

@tool("Volatility Calculator")
def calculate_volatility(ticker: str) -> dict:
    """
    Calculates simulated volatility metrics for a stock.
    Returns volatility percentage and risk assessment.
    """
    logging.info(f"Tool Call: calculate_volatility for ticker '{ticker}'")
    
    # Simulate different volatility levels for different stocks
    volatility_data = {
        "AAPL": 14.5,
        "GOOGL": 18.2,
        "MSFT": 12.8,
    }
    
    volatility = volatility_data.get(ticker.upper(), 15.0 + random.uniform(-3, 3))
    
    # Categorize risk
    if volatility < 12:
        risk_level = "low"
    elif volatility < 18:
        risk_level = "moderate"
    else:
        risk_level = "high"
    
    return {
        "volatility_percent": round(volatility, 2),
        "risk_level": risk_level,
        "assessment": f"{risk_level.title()} risk profile"
    }

@tool("Price Alert Checker")
def check_price_alert(ticker: str, threshold: float) -> dict:
    """
    Checks if a stock price has crossed a specified threshold.
    Returns alert status and comparison details.
    """
    logging.info(f"Tool Call: check_price_alert for {ticker} at threshold ${threshold}")
    
    current_price = get_stock_price(ticker)
    
    alert_triggered = current_price < threshold
    difference = threshold - current_price
    percentage_diff = (difference / threshold) * 100
    
    return {
        "ticker": ticker.upper(),
        "current_price": current_price,
        "threshold": threshold,
        "alert_triggered": alert_triggered,
        "difference": round(difference, 2),
        "percentage_below": round(percentage_diff, 2) if alert_triggered else 0,
        "message": f"Price ${current_price} is {'below' if alert_triggered else 'above'} threshold ${threshold}"
    }

#%% [python]
# Create specialized agents for complex workflow

market_monitor = Agent(
    role='Real-time Market Data Specialist',
    goal='Monitor stock prices and identify alert conditions',
    backstory="""You are a vigilant market data analyst who tracks real-time 
    price movements and identifies when stocks hit key price thresholds. You 
    provide clear, factual price data.""",
    verbose=False,
    tools=[get_stock_price, check_price_alert],
    allow_delegation=False
)

news_analyst = Agent(
    role='Financial News Sentiment Analyst',
    goal='Analyze market sentiment from news headlines',
    backstory="""You are an expert at reading market news and gauging investor 
    sentiment. You fetch relevant news and provide clear sentiment analysis that 
    helps inform trading decisions.""",
    verbose=False,
    tools=[fetch_news_headlines, analyze_sentiment],
    allow_delegation=False
)

risk_assessor = Agent(
    role='Risk Management Specialist',
    goal='Evaluate volatility and risk metrics for stocks',
    backstory="""You are a quantitative analyst specializing in risk assessment. 
    You calculate volatility metrics and provide clear risk evaluations to help 
    guide investment decisions.""",
    verbose=False,
    tools=[calculate_volatility],
    allow_delegation=False
)

trading_strategist = Agent(
    role='Senior Trading Strategist',
    goal='Synthesize market data, sentiment, and risk to make trading recommendations',
    backstory="""You are a veteran trading strategist with 20 years of experience. 
    You synthesize inputs from market data, news sentiment, and risk analysis to 
    provide clear BUY, HOLD, or SELL recommendations with detailed reasoning.""",
    verbose=False,
    tools=[],  # No tools - synthesizes others' work
    allow_delegation=False
)

#%% [python]
# Define tasks with dependencies

price_monitoring_task = Task(
    description="""Monitor the current price for AAPL and check if it has dropped 
    below the $180 threshold. Provide the current price and alert status.""",
    expected_output="""A clear report with current price, threshold comparison, 
    and whether a buying opportunity alert has been triggered.""",
    agent=market_monitor
)

sentiment_analysis_task = Task(
    description="""Fetch recent news headlines for AAPL and analyze the overall 
    sentiment. Provide sentiment scores and interpretation.""",
    expected_output="""A sentiment report showing positive/negative/neutral counts 
    and overall sentiment interpretation.""",
    agent=news_analyst
)

risk_assessment_task = Task(
    description="""Calculate the volatility and risk metrics for AAPL. Provide 
    a risk assessment.""",
    expected_output="""A risk report with volatility percentage and risk level 
    categorization (low/moderate/high).""",
    agent=risk_assessor
)

trading_recommendation_task = Task(
    description="""Based on the price monitoring, sentiment analysis, and risk 
    assessment, provide a trading recommendation for AAPL. Should we BUY, HOLD, 
    or SELL? Provide clear reasoning that incorporates all three inputs.""",
    expected_output="""A single clear recommendation (BUY/HOLD/SELL) with a 
    detailed paragraph explaining the rationale based on price levels, market 
    sentiment, and risk assessment.""",
    agent=trading_strategist,
    context=[price_monitoring_task, sentiment_analysis_task, risk_assessment_task]
)

#%% [python]
# Create the complex crew

trading_analysis_crew = Crew(
    agents=[market_monitor, news_analyst, risk_assessor, trading_strategist],
    tasks=[
        price_monitoring_task, 
        sentiment_analysis_task, 
        risk_assessment_task, 
        trading_recommendation_task
    ],
    process=Process.sequential,
    verbose=False
)

# Visualize the workflow
print("Complex Trading Analysis Workflow:")
print("="*80)
display_crew_workflow(trading_analysis_crew)

#%% [markdown]
"""
Now let's execute the workflow and see the results:
"""

#%% [python]
# Execute the trading analysis
result = trading_analysis_crew.kickoff()
print("\n" + "="*80)
print("TRADING ANALYSIS RESULT:")
print("="*80)
print(result)

