import streamlit as st
from langchain.schema import HumanMessage, AIMessage, LLMResult
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
import hashlib
import os
import json
import torch

# Configuração da página - deve ser o primeiro comando do Streamlit
st.set_page_config(page_title="ChatWithLLM", page_icon="👍")
st.title("ChatWithLLM :computer:")

# Utilitário para calcular o hash de um arquivo
def compute_file_hash(filepath):
    """Computa um hash para um arquivo dado."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Carregar e salvar hashes de documentos
HASHES_FILE = "hashes.json"

def load_document_hashes():
    """Carrega os hashes dos documentos de um arquivo."""
    if os.path.exists(HASHES_FILE):
        with open(HASHES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_document_hashes(hashes):
    """Salva os hashes dos documentos em um arquivo."""
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f)

# Carrega os hashes dos documentos na inicialização
document_hashes = load_document_hashes()

# Verifica se o diretório do vector store existe e não está vazio
persist_directory = 'chromadb'
vectorstore_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)

if not vectorstore_exists:
    st.write("Vector store não encontrado ou vazio. Reprocessando documentos.")
    document_hashes = {}  # Reinicia os hashes dos documentos para forçar o re-embedding
    save_document_hashes(document_hashes)  # Atualiza o arquivo hashes.json

# Função para adicionar metadados aos documentos
def load_documents_with_metadata(filepath, source_name):
    pdf_loader = PyPDFLoader(filepath)
    documents = pdf_loader.load()
    for doc in documents:
        doc.metadata["source"] = source_name
    return documents

# Função para listar os modelos disponíveis no Ollama
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
    print("Modelos disponíveis:", model_names)
else:
    print("Nenhum modelo encontrado ou servidor indisponível.")

# Inicialização do estado da sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_option" not in st.session_state:
    st.session_state.model_option = None

# Inicializa o modelo de embedding e o vector store
if 'embedding_model' not in st.session_state:
    # Verifica se a GPU está disponível, senão usa a CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Usando dispositivo: {device}")

    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

if 'vector_store' not in st.session_state:
    new_documents = []

    try:
        # Tenta carregar o vector store
        st.session_state.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=st.session_state.embedding_model
        )
        # Verifica se o vector store está realmente populado
        if not st.session_state.vector_store._collection.count():
            raise ValueError("Vector store está vazio.")
        st.write("Vector store carregado a partir do diretório de persistência.")
    except Exception as e:
        st.write(f"Nenhum vector store válido encontrado: {e}")
        st.session_state.vector_store = None

    # Processa múltiplos documentos PDF
    documents_path = 'documents'  # Pasta contendo os arquivos PDF
    for filename in os.listdir(documents_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(documents_path, filename)
            document_hash = compute_file_hash(file_path)

            if document_hash in document_hashes:
                st.write(f"Documento '{filename}' já processado. Pulando embedding...")
            else:
                # Determina a fonte com base no nome do arquivo
                if "DSM" in filename or "dsm" in filename:
                    source_name = "DSM"
                elif "CID" in filename or "cid" in filename:
                    source_name = "CID"
                else:
                    source_name = "Unknown"

                # Carrega documentos com metadados
                pdf_documents = load_documents_with_metadata(file_path, source_name)
                new_documents.extend(pdf_documents)

                # Salva o hash do documento processado
                document_hashes[document_hash] = filename
                st.write(f"Documento '{filename}' adicionado para embedding.")

    # Salva os hashes dos documentos atualizados
    save_document_hashes(document_hashes)

    # Adiciona novos documentos ao vector store, se houver
    if new_documents:
        if st.session_state.vector_store:
            st.session_state.vector_store.add_documents(new_documents)
            st.write("Novos documentos embedded e adicionados ao vector store existente.")
        else:
            st.session_state.vector_store = Chroma.from_documents(
                new_documents,
                st.session_state.embedding_model,
                persist_directory=persist_directory
            )
            st.write("Vector store criado e novos documentos embedded.")
    else:
        if st.session_state.vector_store is None:
            st.error("Nenhum documento encontrado para embedding e o vector store está vazio.")
            st.stop()

# Callback Handler do Streamlit
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

    def on_llm_end(self, response: LLMResult, **kwargs):
        self.container.markdown(self.text)

# Função para obter documentos balanceados com limiar de distância
def get_balanced_documents(query, max_docs_per_source=2, max_distance=0.5):
    # Recupere documentos com suas pontuações de distância
    results = st.session_state.vector_store.similarity_search_with_score(query, k=50)
    
    # Filtrar resultados com base no limiar de distância
    filtered_results = [(doc, distance) for doc, distance in results if distance <= max_distance]
    
    # Agrupar documentos por fonte
    grouped_docs = {"DSM": [], "CID": [], "Unknown": []}
    for doc, distance in filtered_results:
        source = doc.metadata.get("source", "Unknown")
        grouped_docs.setdefault(source, []).append((doc, distance))
    
    # Limitar o número de documentos por fonte
    balanced_docs = []
    for source in grouped_docs:
        docs_with_distances = grouped_docs[source][:max_docs_per_source]
        balanced_docs.extend([doc for doc, _ in docs_with_distances])
    
    return balanced_docs

# Função para obter a resposta do modelo
def get_response(query, chat_history, model_option, max_distance):
    # Recupere documentos balanceados com base no limiar de distância
    balanced_docs = get_balanced_documents(query, max_distance=max_distance)
    
    if not balanced_docs:
        st.write("Nenhum documento relevante encontrado com o limiar de distância especificado.")
        return "Desculpe, não encontrei informações relevantes para responder à sua pergunta."

    # Log dos documentos recuperados
    st.write("Documentos recuperados com base no limiar de similaridade:")
    for idx, doc in enumerate(balanced_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        st.write(f"Documento {idx} (Fonte: {source}):")
        st.write(f"Conteúdo da Página:\n{doc.page_content}")
        st.write(f"Metadados: {doc.metadata}")
    
    # Combine o texto dos documentos recuperados
    retrieved_text = "\n\n".join([doc.page_content for doc in balanced_docs])
    
    # Crie o prompt template
    template = """
    Use os seguintes documentos recuperados para responder à pergunta.
    Se a resposta não estiver contida nos documentos abaixo, responda que você não sabe.

    Documentos Recuperados:
    {retrieved_text}

    Pergunta:
    {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_option, temperature=0, streaming=True)
    chain = LLMChain(prompt=prompt, llm=llm)

    # Crie um placeholder no Streamlit
    placeholder = st.empty()
    callback_handler = StreamlitCallbackHandler(placeholder)

    # Execute a cadeia com o callback handler
    chain.run(
        {
            "retrieved_text": retrieved_text,
            "user_question": query
        },
        callbacks=[callback_handler]
    )

    return callback_handler.text

# Função para exibir o histórico de conversa
def display_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Usuário"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

# Sidebar
with st.sidebar:
    models = get_models()  # Certifique-se de que esta função está definida
    st.header("Parâmetros")
    st.session_state.model_option = st.selectbox(
        "Escolha um modelo", models, index=0, help="Modelos disponíveis no Ollama"
    )
    if st.session_state.model_option:
        st.write("Você selecionou: ", st.session_state.model_option, ":white_check_mark:")
    
    st.header("Ajustes de Similaridade")
    max_distance = st.slider(
        "Limiar de Distância Máxima (Similaridade)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Valores menores retornam documentos mais semelhantes."
    )

if not st.session_state.model_option:
    st.warning("Selecione um modelo para começar a conversar.")
    st.stop()

# Exibir histórico de conversa
display_chat_history()

# Entrada do usuário
user_query = st.chat_input("Digite sua pergunta aqui")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Usuário"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(
            user_query,
            st.session_state.chat_history,
            st.session_state.model_option,
            max_distance
        )
        st.session_state.chat_history.append(AIMessage(content=response))
