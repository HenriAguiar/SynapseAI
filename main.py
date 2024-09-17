from api_ollama import get_models

import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

# configurações da página
st.set_page_config(page_title="ChatWithLLM", page_icon="👍")

st.title("ChatWithLLM :computer:")

# inicialização do estado da sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_option" not in st.session_state:
    st.session_state.model_option = None


# função para obter a resposta do modelo
def get_response(query, chat_history, model_option):
    template = """
    Considere o histórico da conversa abaixo, caso esse exista, e responda à pergunta de forma direta.

    Histórico da conversa:
    {chat_history}

    Pergunta:
    {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_option, temperature=0) 
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })


# exibição do histórico do chat
def display_chat_history():
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

# sidebar
with st.sidebar:
    models = get_models()
    st.header("Parâmetros")
    st.session_state.model_option = st.selectbox("Escolha um modelo", models, index=None, placeholder="Disponível no Ollama")
    if st.session_state.model_option:
        st.write("Você escolheu: ", st.session_state.model_option, ":white_check_mark:")

if not st.session_state.model_option:
    st.warning("Selecione um modelo para começar a conversar.")
    st.stop()

# exibe histórico
display_chat_history()

# entrada do usuário
user_query = st.chat_input("Digite sua pergunta aqui")
if user_query:
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("User"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history, st.session_state.model_option))
        st.session_state.chat_history.append(AIMessage(ai_response))

