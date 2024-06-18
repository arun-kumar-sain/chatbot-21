from dotenv import load_dotenv
import os
import streamlit as st
import google.generativeai as genai
from langchain_helper import get_qa_chain, create_vector_db

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.title("Homeopathy CHATBOT 🌱")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Button to create the knowledgebase
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

# Input field for the user question
input_question = st.text_input("Input: ", key="input")

# Button to submit the question
submit = st.button("Ask the question")

if submit and input_question:
    # Get response from Gemini model
    try:
        gemini_response = get_gemini_response(input_question)
        st.session_state['chat_history'].append(("You", input_question))

        st.subheader("The Response is")
        valid_response = False
        for chunk in gemini_response:
            if hasattr(chunk, 'text'):
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))
                valid_response = True
        
        if not valid_response:
            st.error("No valid response received from the Gemini model.")
        
    except ValueError as e:
        st.error(f"An error occurred with the Gemini model response: {e}")

    # Get response from the QA chain
    chain = get_qa_chain()
    try:
        chain_response = chain( input_question)
        # st.write("Chain response structure:", chain_response)
        if "result" in chain_response:
            st.header("QA Chain Answer")
            st.write(chain_response["result"])
            st.session_state['chat_history'].append(("QA Chain Bot", chain_response["result"]))
        else:
            st.error("No result found in the QA chain response.")
    except IndexError as e:
        st.error(f"An error occurred: {e}")
        # st.error("It seems the QA chain returned an unexpected response structure.")

# Display chat history
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
