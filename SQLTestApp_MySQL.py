import streamlit as st
import pandas as pd

from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine

from langchain import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.callbacks.base import BaseCallbackHandler

mysql_uri = f"mysql+mysqlconnector://root:{st.secrets['MySQL']['password']}@localhost:3306/test_1"

#Functions for PLotting map
def modify_query(original_query):
    # Find the position of the FROM keyword
    select_index = original_query.upper().find("SELECT ")
    
    if select_index == -1:
        raise ValueError("No FROM keyword found in the query.")
    
    after_select = original_query[select_index:]
    after_select = after_select.replace('SELECT', ',')
    
    # Add select coordinate columns to table
    select_star_query = "SELECT locations.longitude, locations.latitude " + after_select
    
    return select_star_query

def plot_locations_on_map(locations):
    # Create a folium map centered around the geographic center of the US
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    # Add each location as a red dot on the map
    for _, location in locations.iterrows():
        folium.Marker(
            icon=folium.Icon(color="red", icon = "pushpin"),
            location=[float(location['latitude']), float(location['longitude'])],
            popup=folium.Popup(f"City: {location['city']}", parse_html=True),
            tooltip=location['city']
        ).add_to(m)
        
    return m

def location_map(q, connection):
    df = pd.read_sql_query(q, connection)
    st.write(df)
    if 'longitude' and 'latitude' in df:
        return plot_locations_on_map(df)
    else:
        return None

# LLM Tools
@tool
def bargraph(query: str) -> str:
    """Plots a bargraph from provided dataframe.

    Args:
        query: str
    """
    engine = create_engine(mysql_uri)
    with engine.connect() as conn:
        df = pd.read_sql(sql=query, con=conn.connection)
    st.write(df)
    df = df.set_index(df.columns[0])
    fig, ax = plt.subplots()
    df.plot(kind='bar', title="Generated Plot", ax=ax)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    return "Success"

@tool
def piegraph(query: str) -> str:
    """Plots a piegraph from provided dataframe.

    Args:
        query: str
    """
    engine = create_engine(mysql_uri)
    with engine.connect() as conn:
        df = pd.read_sql(sql=query, con=conn.connection)
    st.write(df)
    df = df.set_index(df.columns[0])
    fig = df.plot.pie(subplots=True)
    st.pyplot(fig)
    return "Success"

@tool
def linegraph(query: str) -> str:
    """Plots a linegraph from provided dataframe.

    Args:
        query: str
    """
    engine = create_engine(mysql_uri)
    with engine.connect() as conn:
        df = pd.read_sql(sql=query, con=conn.connection)
    st.write(df)
    df = df.set_index(df.columns[0])
    fig, ax = plt.subplots()
    df.plot.line(ax=ax)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    return "Success"


@tool
def mapper(query: str) -> str:
    """Plots a map showing locations.
    
    Args:
        query: str
    """
    engine = create_engine(mysql_uri)
    with engine.connect() as conn:
        map_plot = location_map(query, conn.connection)
    try:
        st_folium(map_plot,returned_objects=[])
    except:
        mod_query = modify_query(query)
        engine = create_engine(mysql_uri)
        with engine.connect() as conn:
            map_plot = location_map(mod_query, conn.connection)
        st_folium(map_plot,returned_objects=[])
    return 'Success'

tools = [bargraph, mapper, piegraph, linegraph]

def connect_to_db(mysql_uri):
    db = SQLDatabase.from_uri(mysql_uri,sample_rows_in_table_info=1, include_tables=['locations', 'orders'])
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

def create_structured_agent( 
    db,
    agent_type=AgentType.OPENAI_FUNCTIONS, #agent_type depracated, from old article
    temperature=0,
    model="gpt-3.5-turbo-0125",
 ):
    
    llm = ChatOpenAI(temperature=temperature, model=model, openai_api_key = st.secrets["OpenAI_key"],)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
 
    return create_sql_agent(
        llm = llm,
        toolkit = toolkit,
        agent_type = "openai-tools", 
        verbose=True
    )

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
    structured_agent_db = create_structured_agent(db)
    query, result = run_structured_queries(structured_agent_db, prompt)
    return query, result

# Examples to inject to template for few-shot prompting
examples = [
    HumanMessage(
        "The total order amounts for the warehouses in New York and Chicago are $450.00 and $440.00, respectively."
    ),
    AIMessage(
        "",
        name='example_assistant',
        tool_calls=[
            {"name": "bargraph", "args": "", "id": "1"}
        ]
    ),
    HumanMessage(
        "Order 1 belongs to New York, order 2 belongs to Los Angeles, and order 5 belongs to Miami."
    ),
    AIMessage(
        "",
        name='example_assistant',
        tool_calls=[
            {"name": "mapper", "args": "", "id": "1"}
        ]
    ),
    HumanMessage(
        "The proportion of the sales from Wareshouse A and B is 20% and 80% respectively."
    ),
    AIMessage(
        "",
        name='example_assistant',
        tool_calls=[
            {"name": "piegraph", "args": "", "id": "1"}
        ]
    ),
    HumanMessage(
        "The change in sales over time are as follows"
    ),
    AIMessage(
        "",
        name='example_assistant',
        tool_calls=[
            {"name": "linegraph", "args": "", "id": "1"}
        ]
    ),
]

# Bind tools to LLM (Plotting agent)
llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["OpenAI_key"])
llm_with_tools = llm.bind_tools(tools)

st.title("VizualDB Agent Demo")
st.divider()
text_input = st.text_input("Enter your query")
if text_input:
    # SQL Agent generate SQL query based on user input
    db = connect_to_db(mysql_uri)
    query, result = sql_query(text_input)
    st.write('LLM Generated SQL Query: \n\n' + query['query'] + '\n\n')
    st.write('Result: ' + result["output"])
    
    # Prompt for Plotting Agent to decide plotting tool to use
    tool_call_list = llm_with_tools.invoke(f"""Decide which plotting tool you should use to visualize a result. 
    - If result is about places, plot a map showing location of the place. 
    - If result is categorical, plot a bargraph. 
    - If result is continuous (time series), plot a linegraph.
    You must choose one tool. 
    Result: {result}
                        
    Here are some examples:
    {examples}""").tool_calls

    # Extract tool chosen by LLM from toolcall and run respective function
    print(tool_call_list)
    tool_call = tool_call_list[0]
    print(tool_call)
    if tool_call["name"].lower() == 'bargraph':
        bargraph(query['query'])
    if tool_call["name"].lower() == 'piegraph':
        piegraph(query['query'])
    if tool_call["name"].lower() == 'mapper':
        mapper(query['query'])
    if tool_call["name"].lower() == 'linegraph':
        linegraph(query['query'])
    