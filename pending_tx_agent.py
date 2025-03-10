import asyncio
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import ChatPromptTemplate
import pathlib
import json

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Get the absolute path to the CSV file
current_dir = pathlib.Path(__file__).parent.resolve()
csv_path = current_dir.parent / "docs" / "pending_tx.csv"

# Load the data with correct path
df = pd.read_csv(csv_path)

# Create the pandas agent
agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True,
    allow_dangerous_code=True
)

def get_pending_tx_details(query: str) -> str:
    """
    Query the pending transactions dataframe using natural language
    """
    try:
        # Parse the query string directly if it contains QUERY_TYPE and CUSTOMER_ID
        if "QUERY_TYPE:" in query and "CUSTOMER_ID:" in query:
            lines = query.strip().split("\n")
            query_info = {"query_type": None, "filters": {}}
            
            for line in lines:
                if line.startswith("QUERY_TYPE:"):
                    query_info["query_type"] = line.replace("QUERY_TYPE:", "").strip().lower()
                elif line.startswith("CUSTOMER_ID:"):
                    customer_id = line.replace("CUSTOMER_ID:", "").strip()
                    if customer_id and customer_id != "<customer_id from step 1>":
                        query_info["filters"]["customer_id"] = customer_id
                elif line.startswith("DATE_RANGE:"):
                    date_range = line.replace("DATE_RANGE:", "").strip()
                    if date_range and "-" in date_range:
                        start_date, end_date = date_range.split("-", 1)
                        query_info["filters"]["date_range"] = (start_date.strip(), end_date.strip())
                elif line.startswith("AMOUNT_RANGE:"):
                    amount_range = line.replace("AMOUNT_RANGE:", "").strip()
                    if amount_range and "-" in amount_range:
                        min_amount, max_amount = amount_range.split("-")
                        query_info["filters"]["amount_range"] = (float(min_amount), float(max_amount))
        else:
            # Create a prompt for the LLM to understand the query
            query_analyzer_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at analyzing transaction queries.
                The dataframe has these columns:
                - pending_tx_id: Transaction ID
                - customer_id: Customer ID (e.g., C001, C002, etc.)
                - pending_date: Date of the pending transaction (YYYY-MM-DD)
                - pending_amount: Amount of the transaction in GBP
                
                Format your response exactly as:
                QUERY_TYPE: total|list|specific
                CUSTOMER_ID: <customer_id>
                DATE_RANGE: <start_date>-<end_date> (optional)
                AMOUNT_RANGE: <min_amount>-<max_amount> (optional)
                
                Example:
                QUERY_TYPE: total
                CUSTOMER_ID: C004
                DATE_RANGE: 2025-02-01-2025-03-01
                AMOUNT_RANGE: 500-1000
                """),
                ("human", "{question}")
            ])
            
            # Get query analysis from LLM
            response = llm.invoke(
                query_analyzer_prompt.format(question=query)
            )
            
            # Parse the response
            lines = response.content.strip().split("\n")
            query_info = {"query_type": None, "filters": {}}
            
            for line in lines:
                if line.startswith("QUERY_TYPE:"):
                    query_info["query_type"] = line.replace("QUERY_TYPE:", "").strip().lower()
                elif line.startswith("CUSTOMER_ID:"):
                    customer_id = line.replace("CUSTOMER_ID:", "").strip()
                    if customer_id and customer_id != "<customer_id>":
                        query_info["filters"]["customer_id"] = customer_id
                elif line.startswith("DATE_RANGE:"):
                    date_range = line.replace("DATE_RANGE:", "").strip()
                    if date_range and "-" in date_range:
                        start_date, end_date = date_range.split("-", 1)
                        query_info["filters"]["date_range"] = (start_date.strip(), end_date.strip())
                elif line.startswith("AMOUNT_RANGE:"):
                    amount_range = line.replace("AMOUNT_RANGE:", "").strip()
                    if amount_range and "-" in amount_range:
                        min_amount, max_amount = amount_range.split("-")
                        query_info["filters"]["amount_range"] = (float(min_amount), float(max_amount))
        
        # Apply filters
        filtered_df = df.copy()
        filters = query_info["filters"]
        
        if filters.get("customer_id"):
            filtered_df = filtered_df[filtered_df["customer_id"] == filters["customer_id"]]
        
        if filters.get("date_range"):
            start_date, end_date = filters["date_range"]
            filtered_df = filtered_df[
                (filtered_df["pending_date"] >= start_date) &
                (filtered_df["pending_date"] <= end_date)
            ]
            
        if filters.get("amount_range"):
            min_amount, max_amount = filters["amount_range"]
            filtered_df = filtered_df[
                (filtered_df["pending_amount"] >= min_amount) &
                (filtered_df["pending_amount"] <= max_amount)
            ]
            
        if filtered_df.empty:
            return "No transactions found matching your criteria."
            
        # Format response based on query type
        if query_info["query_type"] == "total":
            total = filtered_df["pending_amount"].sum()
            return f"Total amount: £{total:.2f}"
            
        elif query_info["query_type"] == "list":
            response_parts = ["Pending transactions:"]
            for _, row in filtered_df.iterrows():
                response_parts.append(f"- {row['pending_date']}: £{row['pending_amount']:.2f}")
            
            if len(filtered_df) > 1:
                total = filtered_df["pending_amount"].sum()
                response_parts.append(f"\nTotal: £{total:.2f}")
                
            return "\n".join(response_parts)
            
        else:  # specific transaction details
            return filtered_df.to_string(index=False)
                
    except Exception as e:
        return f"Error processing pending transactions query: {str(e)}"

# Example usage:
if __name__ == "__main__":
    # Test queries
    test_queries = [
        # Direct query format
        """QUERY_TYPE: total
CUSTOMER_ID: C004""",
        # Natural language query
        "What is the total amount of pending transactions for customer C004?",
        # List query
        """QUERY_TYPE: list
CUSTOMER_ID: C004"""
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("Result:", get_pending_tx_details(query))
        print("-" * 80) 