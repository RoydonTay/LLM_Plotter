import streamlit as st
import pandas as pd

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.callbacks.base import BaseCallbackHandler
from langchain_groq import ChatGroq

# Table info
tables_dict = {

}

snowflake_account = "psai-csdw_usec"
database= "DB_DEV"
schema = "NAVIGATOR_SAMPLE_DATA"
warehouse = "COMPUTE_VWH"
role = "DATA_SCIENTIST"
user = st.secrets['snowflake']['user']
password = st.secrets['snowflake']['password']
snowflake_url = f"snowflake://{user}:{password}@{snowflake_account}/{database}/{schema}?warehouse={warehouse}&role={role}"

def connect_to_db(db_uri):
    db = SQLDatabase.from_uri(db_uri, view_support=True, sample_rows_in_table_info=1, ignore_tables=['shipment_details']) #custom_table_info=tables_dict)
    return db

## This is to get the sql query out as well.
class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = None

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can 
         record it as the final sql"""

        if action.tool == "sql_db_query":
            self.sql_result = action.tool_input

def run_structured_queries(db, query: str):
    query_template = f'''{query} Execute all necessary queries, and always return results to the query, no explanations or apologies please, include coordinates if necessary. Word wrap output every 50 characters.'''
    handler = SQLHandler()

    result1 = db.invoke({"input": query_template},
                        {"callbacks": [handler]})
 
    print("=== Generated query ===")
    sql_queries = handler.sql_result
    print(sql_queries)
    return sql_queries, result1
    
def sql_query(prompt):
    llm = ChatGroq(
    temperature=0,
    model="gemma2-9b-it",
    groq_api_key=st.secrets['GROQ_API_KEY']
)
    structured_agent_db = create_sql_agent(llm=llm, db=db, agent_type="zero-shot-react-description", verbose=True)
    query, result = run_structured_queries(structured_agent_db, prompt)
    return query, result

st.title("SQL Agent Demo")
st.divider()
text_input = st.text_input("Enter your query")
if st.button('Submit', type='primary'):
    if text_input:
        # SQL Agent generate SQL query based on user input
        db = connect_to_db(snowflake_url)
        query, result = sql_query(text_input)
        st.write('LLM Generated SQL Query: \n\n' + query + '\n\n')
        st.write('Result: ' + result["output"])
    else:
        st.warning("Please enter question.")
        