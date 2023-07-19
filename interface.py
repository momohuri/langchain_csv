import streamlit as st
import pandas as pd
import json

from agent import query_agent, create_agent


def decode_response(response: str) -> dict:
    return json.loads(response)


def write_response(response_dict: dict):
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


st.title("Welcome Bears! Start chat with your CSV")

st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload a CSV")

query = st.text_area("Insert your query")

if st.button("Submit Query", type="primary"):
    with st.spinner(text="In progress..."):
        # Create an agent from the CSV file.
        agent = create_agent(data)

        # Query the agent.
        response = query_agent(agent=agent, query=query)

        # Decode the response.
        decoded_response = decode_response(response)

        # Write the response to the Streamlit app.
    write_response(decoded_response)
