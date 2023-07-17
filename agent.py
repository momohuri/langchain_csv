from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

# Setting up the api key
import environ
from langchain.chat_models import ChatOpenAI

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")


def create_agent(filename: str):
    # Create an OpenAI object.
    llm = ChatOpenAI(openai_api_key=API_KEY, model='gpt-4', temperature=0.5)

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, df, verbose=True)


def query_agent(agent, query):
    prompt = (
            """
                SYSTEM:
                For the human query, if it requires drawing a table, reply as follows:
                {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
    
                If the query requires creating a bar chart, reply as follows:
                {"bar": {"columns": ["label1", "label2", "label3", ...], "data": [10,2,30, ...]}}
                
                If the query requires creating a line chart, reply as follows:
                {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
                The length of columns should be equal to the length of data
                
                There can only be two types of chart, "bar" and "line".
                
                If it is just asking a question that requires neither, reply as follows:
                {"answer": "answer"}
                Example:
                {"answer": "The title with the highest rating is 'Gilead'"}
                
                If you do not know the answer, reply as follows:
                {"answer": "I do not know."}
                
                All strings in "columns" list and data list, should be in double quotes,
                
                For example: {"columns": ["industry1", "industry2"], "data": [100, 4000]}
                
                ------------
                Human: 
                """
            + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)
    print(response)
    # Convert the response to a string.
    return response.__str__()
