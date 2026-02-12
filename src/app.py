#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG PDF CHATBOT - Interface Streamlit
======================================
Este módulo implementa a interface de usuário (frontend) para o sistema de 
ChatBot RAG (Retrieval-Augmented Generation) especializado em PDFs.

O aplicativo atua como o controlador principal (Controller), gerenciando a interação
entre o usuário e a classe de backend `ChatBot`.

DEPENDENCIAS
=================
- streamlit
- chatbot
- os, json, datetime

USO
========
$ streamlit run src/app.py


AUTOR: Wellington M Santos
DATA: Fevereiro/2026
"""

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import os
from chatbot import ChatBot
from datetime import datetime
import json

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PDFs ChatBot",
    page_icon="📚",
    layout="centered"
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    st.session_state.stats = {
        "queries": 0,
        "last_query_time": None,
        "query_times": [],
        "last_sources": []
    }


# =========================================================
# HEADER
# =========================================================
with st.container(border=True):
    st.title("📚 PDFs ChatBot")
    st.caption("Converse com seus documentos em um workspace inteligente")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    # ===== UPLOADS =====
    st.subheader("Painel de Controle", divider='red')
    
    uploaded = st.file_uploader(
        "Enviar PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
    
    if uploaded:
        if st.button("Processar", use_container_width=True):
            paths = []
            progress = st.progress(0)
            
            for i, file in enumerate(uploaded):
                path = os.path.join(UPLOAD_DIR, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                paths.append(path)
                progress.progress((i + 1) / len(uploaded))
            
            with st.spinner("Indexando..."):
                if st.session_state.bot:
                    st.session_state.bot.load_documents(paths)
            
            st.success("Documentos prontos!")
            st.rerun()
    
    # ===== LISTA DE PDFs CARREGADOS =====
    with st.expander('PDFs Carregados'):
        if st.session_state.bot:
            docs = st.session_state.bot.list_loaded_documents()
            
            if not docs:
                st.caption("Nenhum documento")
            else:
                for d in docs:
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"**{d['name'][:25]}...**")
                            st.markdown(f"""            
                                            - Págs: {d.get('pages','?')}
                                            - Chunks: {d.get('chunks','?')}
                            """)
                        
                        with col2:
                            if st.button("🗑️", key=f"del_{d['name']}", use_container_width=True):
                                st.session_state.bot.remove_document(d["name"])
                                st.rerun()
        else:
            st.caption("🔌 Conecte-se primeiro")
    
    # ===== CONFIGURAÇÕES =====
    if st.session_state.bot:
        with st.expander('Configurações'):
            conf = st.session_state.bot.get_config()
            
            available_models = [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "moonshotai/kimi-k2-instruct",
                "openai/gpt-oss-20b",
                "qwen/qwen3-32b"
            ]
            
            try:
                idx = available_models.index(conf["model"])
            except ValueError:
                idx = 0
            
            model = st.selectbox("Modelo LLM", available_models, index=idx)
            
            temp = st.slider(
                "Temperatura",
                0.0,
                2.0,
                float(conf["temperature"]),
                0.1
            )
                                    
            max_tokens = st.slider(
                    "Max Tokens",
                    256,
                    2048,
                    int(conf["max_tokens"]),
                    256
                )
                
            k = st.slider(
                    "Top K (chunks retornados)",
                    1,
                    10,
                    int(conf["retriever_k"]),
                    1
                )
                
            fetch_k = st.slider(
                    "Fetch K (candidatos)",
                    3,
                    20,
                    int(conf["retriever_fetch_k"]),
                    1
                )
                                
            max_hist = st.slider(
                    "Max mensagens",
                    4,
                    50,
                    int(conf["max_history_messages"]),
                    2
                )
                
            st.caption("⚠️ _Configs de embeddings e chunking requerem reindexação_")
            
            # salvar
            if st.button("💾 Salvar Configurações", use_container_width=True):
                res = st.session_state.bot.update_config(
                    model=model,
                    temperature=temp,
                    max_tokens=max_tokens,
                    retriever_k=k,
                    retriever_fetch_k=fetch_k,
                    max_history_messages=max_hist
                )
                
                if res["success"]:
                    st.success("Configurações aplicadas!")
                    
                    if res.get("warning"):
                        st.warning(f"⚠️ {res['warning']}")
                else:
                    st.error(f"❌ {res['message']}")
        
        # ===== INDICES =====        
        with st.expander("Gerenciar Índices", expanded=False):
            st.warning("⚠️ **Atenção**: Isso deletará os índices do disco permanentemente!")
            
            conf = st.session_state.bot.get_config()
                    
            if st.button("🗑️ Limpar TODOS os Índices", use_container_width=True):
                import shutil
                from pathlib import Path
                        
                index_dir = Path(conf.get("index_dir", "indices"))
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                    index_dir.mkdir()
                    st.session_state.bot.clear_all()
                    st.success("Índices deletados!")
                    st.rerun()
        
        
        if st.button("🔌 Desconectar API", use_container_width=True):
            st.session_state.bot = None
            st.session_state.api_key = None
            st.session_state.messages = []
            st.rerun()

    # FIM SIDEBAR

# =========================================================
# API KEY GATE
# =========================================================
if st.session_state.bot is None:
    st.subheader("🔑 Conectar API")
    
    api_input = st.text_input("Cole sua API KEY da Groq", type="password")
    
    if api_input:
        try:
            with st.spinner("Conectando..."):
                st.session_state.bot = ChatBot(api_key=api_input)
                st.session_state.api_key = api_input
            
            st.success("✅ Conectado com sucesso!")
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Erro ao conectar: {e}")
    
    st.info("💡 Insira sua API key para iniciar.")
    st.stop()

bot = st.session_state.bot

# =========================================================
# MAIN
# =========================================================
tab_chat, tab_status, tab_info = st.tabs(
    ["💬 Chat", "⚡ Status", "ℹ️ Info"]
)

# ===== TAB CHAT =====
with tab_chat:
    # with st.container(border=True):
    #     st.subheader("💬 Chat")
    
    docs_loaded = bot.list_loaded_documents()
    has_docs = len(docs_loaded) > 0
    
    if not has_docs:        
        if len(st.session_state.messages) == 0:
            welcome_message = {
                "role": "assistant",
                "content": """Olá! 👋 Sou seu assistente de documentos PDF.

Para começar, você precisa:

1. **Enviar PDFs** → Use a sidebar à esquerda
2. **Processar** → Clique no botão "Processar" 
3. **Conversar** → Faça suas perguntas!

Posso te ajudar a:
- 📖 Extrair informações específicas dos documentos
- 🔍 Buscar trechos relevantes
- 📊 Resumir conteúdos
- ❓ Responder perguntas sobre os PDFs

Estou pronto para começar assim que você carregar seus documentos! 🚀""",
                "sources": []
            }
            st.session_state.messages.append(welcome_message)
    
    # mostrar mensagens
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # mostrar fontes
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("Fontes consultadas", expanded=False):
                    sources_by_doc = {}
                    for doc in msg["sources"]:
                        doc_name = doc.metadata.get("source_file", "?")
                        page = doc.metadata.get("page", "?")
                        score = doc.metadata.get("score", 0)
                        
                        if doc_name not in sources_by_doc:
                            sources_by_doc[doc_name] = []
                        sources_by_doc[doc_name].append((page, score))
                    
                    for doc_name, pages_scores in sources_by_doc.items():
                        pages = [str(p[0]) for p in pages_scores]
                        st.caption(f"**{doc_name}** → páginas: {', '.join(pages[:5])}")
    
    # query
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
        
        # gerar resposta
        with st.chat_message("assistant"):
            import time
            start_time = time.time()
            
            response = st.write_stream(bot.chat_stream(prompt))
            
            elapsed = time.time() - start_time
            
            # fontes
            sources = bot.last_sources if hasattr(bot, 'last_sources') else []
        
        # appendar res com fontes
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "time": elapsed
        })
        
        # up stats
        st.session_state.stats["queries"] += 1
        st.session_state.stats["last_query_time"] = elapsed
        st.session_state.stats["query_times"].append(elapsed)
        st.session_state.stats["last_sources"] = sources
        
        # 20 medições
        if len(st.session_state.stats["query_times"]) > 20:
            st.session_state.stats["query_times"] = st.session_state.stats["query_times"][-20:]
        
        st.rerun()

    
# ===== TAB STATUS =====
with tab_status:        
        stats = bot.get_stats()
        
        # ========== SEÇÃO 1: CONFIGURAÇÃO ATIVA ==========
        st.subheader("Configuração Ativa")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("**Modelo LLM**")
                st.code(stats['model'].split('/')[-1])
            with col2:
                st.caption("**Temperatura**")
                st.code(f"{stats['temperature']}")
            with col3:
                st.caption("**Embeddings**")
                st.code(stats['embeddings'].split('/')[-1])
                
        # ========== SEÇÃO 2: MÉTRICAS DE USO ==========
        st.subheader("Métricas de Uso")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total de Consultas", st.session_state.stats["queries"])
            col2.metric("Mensagens no Histórico", f"{stats['history_messages']}")
            col3.metric("Documentos Ativos", len(bot.list_loaded_documents()))
                
        # ========== SEÇÃO 3: PERFORMANCE ==========
        st.subheader("Performance")
        with st.container(border=True):
            
            col1, col2 = st.columns(2)
            # ultima query 
            with col1:
                if st.session_state.stats["last_query_time"]:
                    st.metric(
                        "Última Consulta",
                        f"{st.session_state.stats['last_query_time']:.2f}s"
                    )
                else:
                    st.metric("Última Consulta", "—")
            
            # avg queries
            with col2:
                if st.session_state.stats["query_times"]:
                    avg_time = sum(st.session_state.stats["query_times"]) / len(st.session_state.stats["query_times"])
                    st.metric(
                        "Média (últimas 20)",
                        f"{avg_time:.2f}s"
                    )
                else:
                    st.metric("Média (últimas 20)", "—")
            
            # chart com tempo de resposta
            if st.session_state.stats["query_times"]:
                st.caption("**Histórico de Tempo de Resposta**")
                st.line_chart(st.session_state.stats["query_times"])
            else:
                st.info("Faça algumas consultas para ver o gráfico de performance")
                    
        # ========== SEÇÃO 4: AÇÕES ==========
        st.subheader("Ações")
        
        col1, col2 = st.columns(2)
        
        # ========== LIMPAR HISTORICO ==========
        with col1:
            if st.button("🗑️ Limpar Histórico", use_container_width=True):
                bot.history = [bot.history[0]]
                st.session_state.messages = []
                st.success("Histórico limpo!")
                st.rerun()
        
        # ========== JSON HISTORICO ==========
        with col2:
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "stats": st.session_state.stats,
                "messages": st.session_state.messages
            }
            
            json_string = json.dumps(export_data, indent=4, ensure_ascii=False, default=str)
            file_name = f"chat_export_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json"
            
            st.download_button(
                label="💾 Baixar Histórico (JSON)",
                data=json_string,
                file_name=file_name,
                mime="application/json",
                use_container_width=True
            )

        # st.caption("💡 _Futuramente: monitoramento de tokens e custo estimado_")

# ===== TAB INFO =====
with tab_info:
    st.markdown("""
    ### PDFs ChatBot
    
    Sistema de RAG (Retrieval-Augmented Generation) que permite conversar 
    com seus documentos PDF de forma inteligente.
    
    **Stack Técnica:**
    - Python
    - LangChain
    - Groq API
    - Streamlit
    - FAISS + HuggingFace Embeddings
    
    **Recursos:**
    - ✅ Upload de múltiplos PDFs
    - ✅ Indexação persistente
    - ✅ Busca semântica híbrida
    - ✅ Histórico contextualizado
    - ✅ Streaming de respostas
    - ✅ Configurações avançadas
    
    ---
    
    **Desenvolvido por:**  
    Wellington M Santos
    
    [LinkedIn](https://www.linkedin.com/in/wellington-moreira-santos/)
    """)