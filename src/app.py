import streamlit as st
import os
import yaml
from chatbot import ChatBot

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PDFs ChatBot",
    page_icon="📚",
    layout="wide"
)

CONFIG_FILE = "config.yaml"
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# CONFIG HELPERS
# =========================================================
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return None


def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f)


# =========================================================
# SESSION STATE
# =========================================================
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "bot" not in st.session_state:
    st.session_state.bot = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = {"queries": 0}


# =========================================================
# HEADER
# =========================================================

st.title("📚 PDFs ChatBot")
st.caption("Converse com seus documentos em um workspace inteligente")
# st.caption('Desenvolvido por Wellington M Santos')

# =========================================================
# API KEY GATE
# =========================================================
if st.session_state.bot is None:

    st.subheader("🔑 Conectar API")

    api_input = st.text_input("Cole sua API KEY da Groq", type="password")

    if api_input:
        try:
            with st.spinner("Conectando..."):
                st.session_state.bot = ChatBot(
                    config=load_config(),
                    api_key=api_input
                )
                st.session_state.api_key = api_input

            st.success("Conectado com sucesso!")
            st.rerun()

        except Exception as e:
            st.error(f"Erro ao conectar: {e}")

    st.info("Insira sua API key para iniciar.")
    st.stop()


bot = st.session_state.bot
col_chat, col_tabs = st.columns([3, 1], gap='medium', border=True)

# ===================== CHAT COLUMN =====================
with col_chat:

        st.subheader("💬 Chat", divider='red', )

        docs_loaded = bot.list_loaded_documents()
        has_docs = len(docs_loaded) > 0

        if not has_docs:
            st.info("⚪ Envie documentos para começar")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input(
            "Pergunte aos seus documentos...",
            disabled=not has_docs
        )

        if prompt:
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = st.write_stream(bot.chat_stream(prompt))

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            st.session_state.stats["queries"] += 1

# ===================== TABS COLUMN =====================
with col_tabs:
    tab_docs, tab_metrics, tab_settings, tab_info = st.tabs(
    ["📂 Documentos", "📊 Métricas", "⚙️ Ajustes", "ℹ️ Info"]
)
    # ===== TAB METRICS ======
    with tab_docs:
        st.subheader("Enviar Documentos")

        uploaded = st.file_uploader(
            "Enviar PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded:
            if st.button("Processar"):

                paths = []
                progress = st.progress(0)

                for i, file in enumerate(uploaded):
                    path = os.path.join(UPLOAD_DIR, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getbuffer())
                    paths.append(path)
                    progress.progress((i + 1) / len(uploaded))

                with st.spinner("Indexando..."):
                    bot.load_documents(paths)

                st.success("Documentos prontos!")
                st.rerun()

        st.divider()

        docs = bot.list_loaded_documents()

        if not docs:
            st.caption("Nenhum documento carregado")
        else:
            for d in docs:
                c1, c2 = st.columns([3, 1])
                c1.caption(f"📄 {d['name']}")
                if c2.button("❌", key=d["name"]):
                    bot.remove_document(d["name"])
                    st.rerun()

    # ===== TAB METRICS =====
    with tab_metrics:

        st.subheader("Métricas de Uso")

        col1, col2 = st.columns(2)

        col1.metric("Consultas", st.session_state.stats["queries"])
        col2.metric("Docs ativos", len(bot.list_loaded_documents()))

        st.caption("Futuramente: tokens, latência, custo, cache hits")

    # ===== TAB SETTINGS =====
    with tab_settings:

        st.subheader("Configurações do Modelo")

        conf = bot.get_config()

        models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768"
        ]

        idx = models.index(conf["model"]) if conf["model"] in models else 0

        model = st.selectbox("Modelo", models, index=idx)
        temp = st.slider("Temperatura", 0.0, 2.0, float(conf["temperature"]), 0.1)
        k = st.slider("Top K", 1, 10, int(conf["retriever_k"]))

        if st.button("Salvar"):
            res = bot.update_config(model=model, temperature=temp, retriever_k=k)
            if res["success"]:
                save_config(bot.get_config())
                st.success("Configurações salvas")

        st.divider()

        if st.button("Desconectar API"):
            st.session_state.bot = None
            st.session_state.api_key = None
            st.rerun()

    # ===== TAB INFO =====
    with tab_info:
        st.write('Este é um exemplo de chatbot com RAG (Busca com contexto) que neste caso buscará informações solicitadas em fontes fornecidas (arquivos pdfs)')
        st.markdown("""Stack:
                    
        Python
        LangChain
        Groq API
        Streamlit""")
        
        st.markdown("Desenvolvido por Wellington M Santos")
        st.markdown("Linkedin: [in/wellington-moreira-santos](https://www.linkedin.com/in/wellington-moreira-santos/)")