import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Import the weather functions from main.py
from main import get_one_day_forecast, get_week_forecast

# Set up OpenAI API key using environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Define the LLM model
llm = ChatOpenAI(model="gpt-4-0125-preview")

# Define weather tools
def get_weather_current():
    """Use this to get the current weather."""
    return get_one_day_forecast()

def get_weather_week():
    """Use this to get the weather forecast for the next week."""
    return get_week_forecast()

weather_tool = [get_weather_current, get_weather_week]

# Bind the tool
llm_with_weather_tool = llm.bind_tools(weather_tool)

# Define the LLM node
def llm_with_tools_node(state: MessagesState):
    messages = state["messages"]
    return {"messages": [llm_with_weather_tool.invoke(messages)]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("llm", llm_with_tools_node)
builder.add_node("tools", ToolNode(weather_tool))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

graph = builder.compile()

def run_weather_agent(query):
    messages = [HumanMessage(content=query)]
    result = graph.invoke(input={"messages": messages})
    return result['messages'][-1].content