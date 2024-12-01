import streamlit as st
from langchain.schema import HumanMessage, AIMessage, LLMResult
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

# Updated imports
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
import hashlib
import os

# Utility to compute file hash
def compute_file_hash(filepath):
    """Compute a hash for a given file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to list available models from Ollama
def get_models():
    try:
        # List available models
        import ollama
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        return model_names
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        # Return fallback models if connection fails
        return ["llama3.1:8b-instruct-q8_0"]

model_names = get_models()

if model_names:
    print("Available models:", model_names)
else:
    print("No models found or server unavailable.")

# Page configuration
st.set_page_config(page_title="ChatWithLLM", page_icon="👍")
st.title("ChatWithLLM :computer:")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_option" not in st.session_state:
    st.session_state.model_option = None
if "document_hashes" not in st.session_state:
    st.session_state.document_hashes = {}

# Initialize the embedding model and vector store
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if 'vector_store' not in st.session_state:
    # Initialize vector store from existing directory if it exists
    persist_directory = 'chromadb'
    new_documents = []

    try:
        # Attempt to load the vector store
        st.session_state.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=st.session_state.embedding_model
        )
        st.write("Vector store loaded from persistence directory.")
    except Exception:
        st.write("No vector store found. Creating a new one...")
        st.session_state.vector_store = None

    # Process multiple PDF documents
    documents_path = 'documents'  # Folder containing PDF files
    for filename in os.listdir(documents_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(documents_path, filename)
            document_hash = compute_file_hash(file_path)

            if document_hash in st.session_state.document_hashes:
                st.write(f"Document '{filename}' already processed. Skipping embedding...")
            else:
                # Load and process the PDF
                pdf_loader = PyPDFLoader(file_path)
                pdf_documents = pdf_loader.load()
                new_documents.extend(pdf_documents)

                # Save the hash of the processed document
                st.session_state.document_hashes[document_hash] = True
                st.write(f"Document '{filename}' added for embedding.")

    # Add new documents to the vector store if any
    if new_documents:
        if st.session_state.vector_store:
            st.session_state.vector_store.add_documents(new_documents)
            st.write("New documents embedded and added to the existing vector store.")
        else:
            st.session_state.vector_store = Chroma.from_documents(
                new_documents,
                st.session_state.embedding_model,
                persist_directory=persist_directory
            )
            st.write("Vector store created and new documents embedded.")

# Streamlit Callback Handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, response: LLMResult, **kwargs):
        self.container.markdown(self.text)

# Function to get response from the model
# Function to get response from the model
def get_response(query, chat_history, model_option):
    # Initialize the retriever
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(query)

    # Log retrieved documents
    st.write("Retrieved the following documents from the vector store:")
    for idx, doc in enumerate(relevant_docs, 1):
        st.write(f"Document {idx}:")
        st.write(f"Page Content:\n{doc.page_content}")
        st.write(f"Metadata: {doc.metadata}")

    # Combine retrieved documents into a single string
    retrieved_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create the prompt template
    template = """
    Use the following retrieved documents to answer the question.
    If the answer is not contained within the documents below, respond that you don't know.

    Retrieved Documents:
    {retrieved_text}

    Question:
    {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_option, temperature=0, streaming=True)
    chain = LLMChain(prompt=prompt, llm=llm)

    # Create a placeholder in Streamlit
    placeholder = st.empty()
    callback_handler = StreamlitCallbackHandler(placeholder)

    # Run the chain with the callback handler
    chain.run(
        {
            "retrieved_text": retrieved_text,
            "user_question": query
        },
        callbacks=[callback_handler]
    )

    return callback_handler.text


# Display chat history
def display_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

# Sidebar
with st.sidebar:
    models = get_models()  # Ensure this function is defined
    st.header("Parameters")
    st.session_state.model_option = st.selectbox("Choose a model", models, index=None, placeholder="Available in Ollama")
    if st.session_state.model_option:
        st.write("You selected: ", st.session_state.model_option, ":white_check_mark:")

if not st.session_state.model_option:
    st.warning("Select a model to start chatting.")
    st.stop()

# Display chat history
display_chat_history()

# User input
user_query = st.chat_input("Type your question here")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("User"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history, st.session_state.model_option)
        st.session_state.chat_history.append(AIMessage(content=response))
