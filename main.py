import streamlit as st
from langchain.schema import HumanMessage, AIMessage, LLMResult
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

# Updated imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.document_loaders import PyPDFLoader  # Import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
def get_models():
    try:
        # Tenta listar os modelos disponíveis
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        return model_names
    except Exception as e:
        print(f"Erro ao conectar ou listar modelos do Ollama: {e}")
        # Retorna modelos manualmente configurados como fallback
        return ["llama3.1:8b-instruct-q8_0"]
# Page configuration
st.set_page_config(page_title="ChatWithLLM", page_icon="👍")
st.title("ChatWithLLM :computer:")

# Session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_option" not in st.session_state:
    st.session_state.model_option = None

# Initialize the embedding model and vector store
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if 'vector_store' not in st.session_state:
    # Load PDF documents
    #pdf_loader = PyPDFLoader('documents\DSM_5_Atualizado_Portugues_DSM_5_TR_Amer.pdf')  # Replace with your PDF file path
    #pdf_documents = pdf_loader.load()

    # If you have multiple PDFs in a directory
    pdf_loader = DirectoryLoader('documents', glob='**/*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = pdf_loader.load()

    # Load text documents if needed
    # text_loader = DirectoryLoader('path/to/your/text_documents', glob='**/*.txt', loader_cls=TextLoader)
    # text_documents = text_loader.load()

    # Combine documents (if you have text documents)
    # documents = pdf_documents + text_documents

    # If only PDFs are used
    documents = pdf_documents

    # Initialize vector store
    persist_directory = 'chromadb'
    st.session_state.vector_store = Chroma.from_documents(
        documents,
        st.session_state.embedding_model,
        persist_directory=persist_directory
    )

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
def get_response(query, chat_history, model_option):
    # Initialize the retriever
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(query)

    # Extract content and metadata
    retrieved_info = []
    for doc in relevant_docs:
        content = doc.page_content
        page_number = doc.metadata.get('page', 'Unknown')
        retrieved_info.append(f"Page {page_number}:\n{content}")

    retrieved_text = "\n\n".join(retrieved_info)

    template = """
    Use the following retrieved documents to answer the question.
    For each piece of information you provide, include the page number it came from.
    If the answer is not contained within the documents below, respond that you don't know.

    Retrieved Documents:
    {retrieved_text}

    Question:
    {user_question}

    Answer (include page numbers):
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
