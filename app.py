import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

# Check if the API key is loaded
if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop the Streamlit app if API key is missing

# Load the dataset
@st.cache_data
def load_data(file_path):
    """
    Loads the HR dataset from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame() # Return an empty DataFrame on error

# Function to send data and prompt to the Mistral AI Chat Completions API
def send_prompt_to_mistral(prompt_text, data_context):
    """
    Sends a prompt and data context to the Mistral AI chat completions API.
    """
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Construct the message payload.
    # We include the data context as part of the system or user message.
    # For large data, consider summarizing or chunking it first.
    messages = [
        {
            "role": "system",
            "content": "You are a highly intelligent HR data analyst. Your task is to answer questions about the provided HR dataset. The dataset is in a text format (CSV converted to string). Analyze the data carefully and provide insightful answers based on the information provided. If a question cannot be answered from the provided data, state that clearly. Be concise and accurate."
        },
        {
            "role": "user",
            "content": f"Here is the HR dataset:\n\n{data_context}\n\nBased on this data, please answer the following question:\n\n{prompt_text}"
        }
    ]

    payload = {
        "model": "mistral-large-latest", # Or other suitable Mistral model
        "messages": messages,
        "temperature": 0.1, # Lower temperature for more deterministic answers
        "max_tokens": 10000 # Adjust as needed for expected response length
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
        return {"error": str(e)}

# Main Streamlit app
def main():
    st.title('HR Dataset Switzerland Q&A with Mistral AI')

    st.markdown("""
    Ask questions about the `hr_dataset_switzerland.csv` file, and Mistral AI will provide insights.
    """)

    # Define the path to your CSV file
    csv_file_path = 'hr_dataset_switzerland.csv'

    # Load data
    data = load_data(csv_file_path)

    if data.empty:
        st.warning("Could not load data. Please ensure 'hr_dataset_switzerland.csv' is in the same directory.")
        return # Exit if data loading failed

    # Convert the DataFrame to a string format for the LLM
    # Using to_csv() as a string ensures proper CSV formatting, including headers
    # For very large files, df.to_markdown() might be more human-readable but less structured for an LLM
    data_as_string = data.to_csv(index=False)

    # Display a warning if the data string is very long (approaching context window limits)
    # This is a rough estimate; actual token count varies.
    if len(data_as_string) > 10000: # Example threshold, adjust based on experimentation
        st.warning(f"The dataset is large ({len(data_as_string)} characters). "
                   "If you encounter errors about context length, consider using a more advanced data chunking/retrieval strategy (RAG).")

    # Text input for user query
    user_input = st.text_area("Enter your question or query here:", height=100,
                              placeholder="e.g., How many rows of data does the file contain? What's the average age of employees?")

    if st.button('Get Answer from Mistral AI'):
        if user_input:
            with st.spinner('Thinking... (Sending data and question to Mistral AI)'):
                # Send the input and the data as context to the API
                response = send_prompt_to_mistral(user_input, data_as_string)

            # Display the API response
            st.subheader('Mistral AI Response')
            if "error" in response:
                st.error(f"Error from API: {response.get('error', 'Unknown error')}")
                if "context_length_exceeded" in str(response.get('error', '')):
                    st.info("The data you sent likely exceeded the model's context window. "
                            "You might need to shorten the CSV data or implement RAG.")
            elif response and response.get('choices'):
                st.write(response['choices'][0]['message']['content'])
            else:
                st.write("No valid response received from the API.")
                st.json(response) # Show the full response for debugging
        else:
            st.write("Please enter a question or query.")

    st.divider() # Separator

    # Display raw data (optional, for user reference)
    with st.expander("View Raw Data"):
        st.dataframe(data)

if __name__ == '__main__':
    main()