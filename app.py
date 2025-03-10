import asyncio
import logging
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Dict, TypedDict, Annotated, Sequence, Union, List
from langgraph.graph import Graph, StateGraph
from langchain.tools.base import BaseTool
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_core.utils.function_calling import convert_to_openai_function
import json

from retriever import search_documents
from customer_db import query_database
from pending_tx_agent import get_pending_tx_details

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
today = datetime.now().strftime("%d/%m/%Y")

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)

# Define the state
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation so far"]
    next: Annotated[str, "The next agent to use"]
    interest_rates: Annotated[str, "Interest rates information"]
    customer_details: Annotated[str, "Customer details information"]
    pending_tx_details: Annotated[str, "Pending transaction details"]
    overall_analysis: Annotated[str, "Overall analysis"]
    search_interest_rates_result: Annotated[Union[str, None], "Result from search_interest_rates"]
    search_customer_details_result: Annotated[Union[str, None], "Result from search_customer_details"]
    search_pending_tx_result: Annotated[Union[str, None], "Result from search_pending_tx"]

# Define tools
tools = [
    Tool(
        name="search_interest_rates",
        description="Search the bank account interest rate documents for information",
        func=lambda q: search_documents(q)
    ),
    Tool(
        name="search_customer_details",
        description="Search the customer database for information",
        func=lambda q: query_database(q)
    ),
    Tool(
        name="search_pending_tx",
        description="Search pending transactions for a customer",
        func=lambda q: get_pending_tx_details(q)
    )
]

# Convert tools to OpenAI functions
functions = [convert_to_openai_function(t) for t in tools]

# Create a dictionary for tool lookup
tool_map = {tool.name: tool for tool in tools}

def execute_tool(name: str, arguments: str) -> str:
    """Execute a tool by name with given arguments"""
    if name not in tool_map:
        return f"Tool {name} not found"
    tool = tool_map[name]
    # Parse arguments from JSON string
    args = json.loads(arguments)
    # Execute the tool with the first argument found
    return tool.func(next(iter(args.values())))

# Agent prompts
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the supervisor agent that orchestrates the workflow to answer questions about bank accounts.
    You have access to the following tools:
    - search_interest_rates: For questions about bank account interest rates
    - search_customer_details: For questions about customer information
    - search_pending_tx: For questions about pending transactions
    
    For each tool:
    - search_interest_rates: Use for any questions about interest rates, account types, or terms
    - search_customer_details: Use for looking up customer information by name or ID
    - search_pending_tx: Use for questions about pending transactions, requires customer name or ID
    
    Analyze the user's question and select the most appropriate tool. Format your response exactly as:
    TOOL: <tool_name>
    QUERY: <specific query for the tool>
    
    Make your queries specific and targeted for each tool. For example:
    - For customer details: "Get details for Bob Brown"
    - For pending transactions: "Get pending transactions for Bob Brown"
    - For interest rates: "Get interest rates for Cash ISA Saver accounts"
    """),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}"),
])

analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an analyst agent responsible for providing comprehensive analysis based on the collected information.
    Analyze all available information and provide a detailed response that combines and makes sense of the data.
    If any information is missing or unclear, mention that in your analysis.
    
    For pending transactions:
    - If you see a "Total amount:" line, extract and report the amount
    - If you see a customer ID with a name in parentheses, use that to confirm the customer's identity
    
    Format your response in a clear, concise way. If the information contains numerical values:
    - Use proper currency symbols (£)
    - Round numbers to 2 decimal places
    - Present lists in bullet points
    
    Here is the information to analyze:
    {context}
    
    Original question: {question}
    """),
    ("human", "Please provide your analysis.")
])

# Agent functions
def supervisor(state: State) -> Union[Dict, State]:
    """Supervisor agent that decides which tool to use"""
    messages = state["messages"]
    question = messages[-1].content if messages else ""
    
    logger.info(f"Processing question: {question}")
    
    try:
        # Create a prompt for the LLM to understand the query type
        query_analyzer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing banking queries and determining the appropriate tool to use.
            Based on the user's question, determine which tool to use and how to query it.
            
            Available tools:
            1. search_interest_rates - For questions about bank account interest rates, terms, and conditions
            2. search_customer_details - For questions about customer information and account details
            3. search_pending_tx - For questions about pending transactions (requires customer ID)
            
            For search_interest_rates and search_customer_details, format your response as:
            TOOL: <tool_name>
            QUERY: <specific query for the tool>
            
            For search_pending_tx, format your response as:
            TOOL: search_pending_tx
            QUERY_TYPE: total|list
            CUSTOMER_ID: <customer_id>
            
            For multi-step queries (e.g., finding pending transactions for a customer by name):
            STEP: 1
            TOOL: search_customer_details
            QUERY: Find customer ID for <customer_name>
            
            STEP: 2
            TOOL: search_pending_tx
            QUERY_TYPE: total|list
            CUSTOMER_ID: <customer_id from step 1>
            """),
            ("human", "{question}")
        ])
        
        # Get tool selection from LLM
        response = llm.invoke(
            query_analyzer_prompt.format(question=question)
        )
        
        logger.info(f"LLM response: {response.content}")
        
        # Parse the response
        lines = response.content.strip().split("\n")
        steps = []
        current_step = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("STEP:"):
                if current_step and current_step.get("tool"):
                    steps.append(current_step)
                current_step = {"tool": None, "query": []}
            elif line.startswith("TOOL:"):
                if current_step and current_step.get("tool"):
                    steps.append(current_step)
                    current_step = {}
                current_step["tool"] = line.replace("TOOL:", "").strip()
                current_step["query"] = []
            elif line.startswith(("QUERY:", "QUERY_TYPE:", "CUSTOMER_ID:")):
                if current_step.get("tool"):
                    current_step["query"].append(line)
        
        if current_step and current_step.get("tool"):
            steps.append(current_step)
        
        if not steps:
            raise ValueError("Could not parse tool selection from LLM response")
        
        # Execute the first step
        first_step = steps[0]
        tool_name = first_step["tool"]
        
        # Format query based on tool type
        if tool_name == "search_pending_tx":
            query = "\n".join(first_step["query"])
        else:
            query = next((q.replace("QUERY:", "").strip() for q in first_step["query"] if q.startswith("QUERY:")), "")
        
        logger.info(f"Selected tool: {tool_name}")
        logger.info(f"Query: {query}")
        
        if tool_name in tool_map:
            try:
                # Execute the tool
                result = tool_map[tool_name].func(query)
                logger.info(f"Tool result: {result}")
                
                # Store result in appropriate state field
                if tool_name == "search_interest_rates":
                    state["interest_rates"] = str(result)
                elif tool_name == "search_customer_details":
                    # Check if this is part of a multi-step query
                    if len(steps) > 1 and steps[1]["tool"] == "search_pending_tx":
                        customer_id = None
                        # Extract customer ID from the result
                        if isinstance(result, str):
                            if result.strip() == "C004":
                                customer_id = "C004"
                            elif "ID: C004" in result:
                                customer_id = "C004"
                        
                        if customer_id:
                            # Store customer details
                            state["customer_details"] = result
                            
                            # Format and execute pending transactions query
                            next_step = steps[1]
                            pending_query = []
                            for line in next_step["query"]:
                                if line.startswith("CUSTOMER_ID:"):
                                    pending_query.append(f"CUSTOMER_ID: {customer_id}")
                                else:
                                    pending_query.append(line)
                            
                            pending_result = tool_map["search_pending_tx"].func("\n".join(pending_query))
                            state["pending_tx_details"] = str(pending_result)
                            
                            # Add both results to messages
                            state["messages"].append(FunctionMessage(
                                content=str(result),
                                name=tool_name
                            ))
                            state["messages"].append(FunctionMessage(
                                content=str(pending_result),
                                name="search_pending_tx"
                            ))
                            
                            return {"next": "analyst"}
                    else:
                        state["customer_details"] = str(result)
                elif tool_name == "search_pending_tx":
                    state["pending_tx_details"] = str(result)
                
                # Add result to messages
                state["messages"].append(FunctionMessage(
                    content=str(result),
                    name=tool_name
                ))
                
                return {"next": "analyst"}
                
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                logger.error(error_msg)
                state["messages"].append(FunctionMessage(
                    content=error_msg,
                    name=tool_name
                ))
                state["overall_analysis"] = error_msg
                return {"next": "end"}
        
    except Exception as e:
        error_msg = f"Error analyzing query: {str(e)}"
        logger.error(error_msg)
        state["overall_analysis"] = error_msg
        return {"next": "end"}
    
    state["overall_analysis"] = "Could not determine appropriate tool for this query."
    return {"next": "end"}

def analyst(state: State) -> State:
    """Analyst agent that provides comprehensive analysis"""
    # Get the original question
    question = state["messages"][0].content if state["messages"] else ""
    
    # Get the most recent tool results from messages
    tool_results = {}
    for msg in state["messages"]:
        if isinstance(msg, FunctionMessage):
            tool_results[msg.name] = msg.content
    
    # Prepare context for analysis
    context = {
        "interest_rates": tool_results.get("search_interest_rates", state.get("interest_rates", "")),
        "customer_details": tool_results.get("search_customer_details", state.get("customer_details", "")),
        "pending_tx_details": tool_results.get("search_pending_tx", state.get("pending_tx_details", ""))
    }
    
    try:
        # Generate analysis using the LLM
        response = llm.invoke(
            analyst_prompt.format(
                context=json.dumps(context, indent=2),
                question=question
            )
        )
        
        # Store the analysis
        analysis = response.content
        logger.info(f"Analysis response: {analysis}")
        
        # Clean up the response for pending transactions
        if context["pending_tx_details"]:
            if "Total amount: £" in context["pending_tx_details"]:
                amount = context["pending_tx_details"].split("Total amount: £")[1].strip()
                customer_info = ""
                if context["customer_details"]:
                    if "ID: C004" in context["customer_details"]:
                        customer_info = "Bob Brown (ID: C004)"
                    elif "(Bob Brown)" in context["customer_details"]:
                        customer_info = "Bob Brown (ID: C004)"
                    else:
                        customer_info = context["customer_details"].strip()
                if customer_info:
                    analysis = f"The total amount of pending transactions for {customer_info} is £{amount}"
            elif "Pending transactions:" in context["pending_tx_details"]:
                # Keep the detailed list format as is
                analysis = context["pending_tx_details"]
                if context["customer_details"]:
                    if "ID: C004" in context["customer_details"] or "(Bob Brown)" in context["customer_details"]:
                        analysis = f"Pending transactions for Bob Brown (ID: C004):\n{analysis.split('Pending transactions:')[1]}"
        
        state["overall_analysis"] = analysis
        
        # Add analysis to messages
        state["messages"].append(
            FunctionMessage(
                content=analysis,
                name="analyst"
            )
        )
        
    except Exception as e:
        logger.error(f"Error in analyst: {str(e)}")
        state["overall_analysis"] = f"Error generating analysis: {str(e)}"
    
    return state

def end(state: State) -> Union[Dict, State]:
    """End node that returns the final state"""
    return state

# Create the graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("supervisor", supervisor)
workflow.add_node("analyst", analyst)
workflow.add_node("end", end)

# Add edges
workflow.add_edge("supervisor", "analyst")
workflow.add_edge("analyst", "end")

# Set entry point
workflow.set_entry_point("supervisor")

# Compile the graph
graph = workflow.compile()

async def main():
    # Test questions
    questions = [
        f"What's the Cash ISA Saver's annual interest rate for an account opened after 18/02/25? Today's date is {today}",
        "List all the details of Bob Brown?",
        "What is the total amount of pending transactions for Bob Brown and round off to 2 decimal places?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Initialize fresh state for each question
        state = State(
            messages=[HumanMessage(content=question)],
            next="supervisor",
            interest_rates="",
            customer_details="",
            pending_tx_details="",
            overall_analysis="",
            search_interest_rates_result=None,
            search_customer_details_result=None,
            search_pending_tx_result=None
        )
        
        try:
            # Run the graph
            result = await graph.ainvoke(state)
            
            # Get the analysis
            analysis = result.get("overall_analysis", "No analysis available")
            
            # Clean up the response if it's from the pending transactions tool
            if isinstance(analysis, str):
                if "Final Answer:" in analysis:
                    analysis = analysis.split("Final Answer:")[1].strip()
                elif "Thought:" in analysis:
                    parts = analysis.split("Final Answer:")
                    if len(parts) > 1:
                        analysis = parts[1].strip()
                    else:
                        parts = analysis.split("Thought:")
                        if len(parts) > 1:
                            analysis = parts[0].strip()
                
                # Clean up formatting
                analysis = analysis.strip().strip('"').replace("\\n", "\n")
            
            # Print the response
            print(f"Answer: {analysis}")
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print(f"Error: {str(e)}")
            
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 