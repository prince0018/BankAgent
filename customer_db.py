from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    inspect,
)
from sqlalchemy import insert, text
from dotenv import load_dotenv
import os
import json

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def create_banking_customer_db():
    """
    Creates and populates a banking customer database if it doesn't exist
    """
    # Set up Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create database
    engine = create_engine("sqlite:///banking_customer.db", future=True)
    
    # Check if the customer table exists
    inspector = inspect(engine)
    if "customer" in inspector.get_table_names():
        return engine
    
    # Create new metadata for the new tables
    metadata_obj = MetaData()
    
    customer_table = Table(
        "customer",
        metadata_obj,
        Column("customer_id", String(16), primary_key=True),
        Column("customer_name", String(50), nullable=False),
        Column("customer_address", String(50), nullable=False),
        Column("customer_phone", String(50)),
        Column("customer_email", String(50)),
        Column("customer_dob", String(50)),
        Column("customer_gender", String(50)),
        Column("customer_nationality", String(50)),
        Column("customer_occupation", String(50)),
        Column("customer_income", String(50)),
        Column("account_balance", String(50)),
    )

    metadata_obj.create_all(engine)

    # Insert data into the customer table
    rows = [
        {"customer_id": "C001", "customer_name": "John Smith", "customer_address": "123 Oak St, London, UK", "customer_phone": "123-456-7890", "customer_email": "john.smith@example.com", "customer_dob": "1980-01-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Engineer", "customer_income": "£60000", "account_balance": "£15000"},
        {"customer_id": "C003", "customer_name": "Alice Johnson", "customer_address": "789 Pine St, Anytown, UK", "customer_phone": "123-456-7892", "customer_email": "alice.johnson@example.com", "customer_dob": "1992-03-20", "customer_gender": "Female", "customer_nationality": "UK", "customer_occupation": "Teacher", "customer_income": "£40000", "account_balance": "£8000"},
        {"customer_id": "C004", "customer_name": "Bob Brown", "customer_address": "123 Main St, Anytown, UK", "customer_phone": "123-456-7893", "customer_email": "bob.brown@example.com", "customer_dob": "1985-05-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Doctor", "customer_income": "£70000", "account_balance": "£25000"},
        {"customer_id": "C005", "customer_name": "Charlie Davis", "customer_address": "456 Oak Ave, Anycity, UK", "customer_phone": "123-456-7894", "customer_email": "charlie.davis@example.com", "customer_dob": "1990-01-01", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Software Engineer", "customer_income": "£50000", "account_balance": "£12000"},
        {"customer_id": "C006", "customer_name": "Tom Johns", "customer_address": "789 Pine St, Anytown, UK", "customer_phone": "123-456-7895", "customer_email": "tom.johns@example.com", "customer_dob": "1992-03-20", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Teacher", "customer_income": "£40000", "account_balance": "£7500"},
        {"customer_id": "C007", "customer_name": "Jane Fonda", "customer_address": "123 Main St, Anytown, UK", "customer_phone": "123-456-7896", "customer_email": "jane.fonda@example.com", "customer_dob": "1985-05-15", "customer_gender": "Male", "customer_nationality": "UK", "customer_occupation": "Doctor", "customer_income": "£70000", "account_balance": "£30000"},
    ]

    for row in rows:
        stmt = insert(customer_table).values(**row)
        with engine.begin() as connection:
            connection.execute(stmt)

    return engine

def query_database(query: str):
    """Execute a natural language query against the database"""
    # Set up Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    
    engine = create_banking_customer_db()
    
    try:
        # Set up SQL query generation prompt
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at converting natural language questions into SQL queries.
            The database has a 'customer' table with these columns:
            - customer_id (VARCHAR)
            - customer_name (VARCHAR)
            - customer_address (VARCHAR)
            - customer_phone (VARCHAR)
            - customer_email (VARCHAR)
            - customer_dob (VARCHAR)
            - customer_gender (VARCHAR)
            - customer_nationality (VARCHAR)
            - customer_occupation (VARCHAR)
            - customer_income (VARCHAR)
            - account_balance (VARCHAR)
            
            For customer name queries, always use LIKE with wildcards:
            - If the query is just a name, use: SELECT * FROM customer WHERE customer_name LIKE :name
            - If the query asks to find a customer ID, use: SELECT customer_id FROM customer WHERE customer_name LIKE :name
            
            Format your response exactly as:
            SQL: <the SQL query>
            PARAMS: <param1>=<value1>, <param2>=<value2>, ...
            
            Example:
            SQL: SELECT * FROM customer WHERE customer_name LIKE :name
            PARAMS: name=%Bob Brown%
            """),
            ("human", "{question}")
        ])
        
        # Get SQL query from LLM
        response = llm.invoke(
            sql_prompt.format(question=query)
        )
        
        # Parse the response
        lines = response.content.strip().split("\n")
        sql = None
        params = {}
        
        for line in lines:
            if line.startswith("SQL:"):
                sql = line.replace("SQL:", "").strip()
            elif line.startswith("PARAMS:"):
                params_str = line.replace("PARAMS:", "").strip()
                if params_str:
                    for param in params_str.split(","):
                        key, value = param.strip().split("=")
                        # Remove any quotes around the value
                        value = value.strip('"\'')
                        # For customer name queries, ensure wildcards are properly added
                        if key.strip() == "name" and "LIKE" in sql:
                            if not value.startswith("%"):
                                value = f"%{value}"
                            if not value.endswith("%"):
                                value = f"{value}%"
                        params[key.strip()] = value
        
        if not sql:
            raise ValueError("No SQL query found in LLM response")
        
        # Execute the query
        with engine.connect() as connection:
            result = connection.execute(text(sql), params)
            rows = result.fetchall()
            
            if not rows:
                return "No customer found with the specified criteria."
            
            # Format the response
            columns = result.keys()
            response_parts = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # If only customer_id was requested, return just that
                if len(columns) == 1 and "customer_id" in row_dict:
                    return row_dict["customer_id"]
                # Otherwise, format the full customer details
                formatted_row = "\n".join([
                    "Customer Details:",
                    "----------------",
                    f"Name: {row_dict['customer_name']}",
                    f"ID: {row_dict['customer_id']}",
                    f"Address: {row_dict['customer_address']}",
                    f"Phone: {row_dict['customer_phone']}",
                    f"Email: {row_dict['customer_email']}",
                    f"Date of Birth: {row_dict['customer_dob']}",
                    f"Gender: {row_dict['customer_gender']}",
                    f"Nationality: {row_dict['customer_nationality']}",
                    f"Occupation: {row_dict['customer_occupation']}",
                    f"Income: {row_dict['customer_income']}",
                    f"Account Balance: {row_dict['account_balance']}"
                ])
                response_parts.append(formatted_row)
            
            return "\n\n".join(response_parts)
            
    except Exception as e:
        return f"Error querying the database: {str(e)}"

# Example usage:
if __name__ == "__main__":
    query = "Show me details of customers who have an account balance over £20000"
    print(query_database(query)) 