# -*- coding: utf-8 -*-
"""
RAGChatBot 
Combina: Persistência automática + Contextualização LCEL + Modo híbrido

Recursos:
- Persistência FAISS automática (índice salvo em disco)
- Contextualização de perguntas com histórico (LCEL)
- Modo híbrido: RAG on/off
- Múltiplos PDFs com merge de índices
"""

################################################################################
# IMPORTS
################################################################################

import os
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union

# LLM 
from langchain_groq import ChatGroq

## core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# pdf loader e vectordb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS

# embeddings e textsplitters
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

################################################################################
# CONFIGURACOES
################################################################################

load_dotenv()

GROQ_MODELS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "llama-3.1-8b": "llama-3.1-8b-instant",
    "kimi-k2": "moonshotai/kimi-k2-instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen3-32b": "qwen/qwen3-32b"
}

DEFAULT_CONFIG = {
    "model": GROQ_MODELS["llama-3.1-8b"],
    "temperature": 0.1,
    "max_tokens": 2048,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
    "retriever_k": 4,
    "retriever_fetch_k": 8,
    "max_history_messages": 10,
    "index_path": "faiss_index",  
}

################################################################################
# CLASSE PRINCIPAL
################################################################################

class RAGChatBot:
    """
    ChatBot RAG com persistência, contextualização e modo híbrido.

    Uso:
        bot = RAGChatBot()
        bot.load_documents(['doc1.pdf', 'doc2.pdf'])
        bot.chat('Qual o tema principal?')
    """

    def __init__(self, config:Dict[str,Any]=None, api_key:str=None):
        """
        Inicializa o chatbot

        Args:
            config: configurações personalizadas (opcional)
            api_key: chave API Groq(opcional, lê do .env se não fornecida)
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.api_key = api_key or os.getenv('GROQ_API_KEY')

        if not self.api_key:
            raise ValueError('GROQ_API_KEY não configurada, use .env ou passe api_key=')

        self.llm = None
        self.history:List[Any] = []
        self.documents_loaded = False
        self.vectorstore = None
        self.embeddings = None
        self.retriever = None
        self.rag_chain = None
        self.contextualized_retriever = None


        print('=== INICIALIZANDO CHATBOT ===')
        self._initialize_llm()
        self._initialize_system_prompt()
        self._initialize_embeddings()
        print('ChatBot pronto!\n')

    def _initialize_llm(self):
        """Inicializa o modelo de linguagem Groq"""
        self.llm = ChatGroq(
            model=self.config['model'],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            api_key=self.api_key
        )
        print(f'LLM: {self.config["model"]} carregado!')

    def _initialize_system_prompt(self):
        """Mensagem fixa do sistema"""
        system_message = SystemMessage(
            content="""Você é um assistente prestativo especializado em responder perguntas com base em documentos fornecidos. 
Seja objetivo, preciso e cite fontes quando possível.""")
        self.history.append(system_message)

    def _initialize_embeddings(self):
        """Inicializa o modelo de embeedings"""
        print(f"Carregando embeddings: {self.config['embeddings_model']}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings_model'],
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )
        print('Embeddings carregados!')
    
    def load_documents(
            self, 
            pdf_paths: Union[str, List[str]], 
            index_path: str = None,
            force_recreate: bool = False
        ) -> bool:
            """
            Carrega PDFs criando ou carregando índice FAISS.
    
            Args:
                pdf_paths: Caminho ou lista de caminhos de PDFs
                index_path: Pasta para salvar/carregar índice (padrão: config['index_path'])
                force_recreate: Se True, recria índice mesmo se existir
    
            Returns:
                True se sucesso, False caso contrário
            """
            if isinstance(pdf_paths, str):
                 pdf_paths = [pdf_paths]
    
            index_path = index_path or self.config['index_path']
    
            print('Carregando {} documento(s)'.format(len(pdf_paths)))
    
            try:
                if not force_recreate and os.path.exists(index_path) and os.path.isdir(index_path): #indice salvo?
                    print('Indice encontrado em {}'.format(index_path))
                    self._load_vectorstore(index_path)
                    print('Indice carregado do disco.')
                else:
                    if force_recreate and os.path.exists(index_path):
                        print('Removendo índice antigo...')
                        shutil.rmtree(index_path)
    
                    all_chunks = self._process_pdfs(pdf_paths)
                    if not all_chunks:
                        print('Nenhum documento processado.')
                        return False
    
                    print('Criando indice FAISS com {} chunks.'.format(len(all_chunks)))
                    self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
                    self.vectorstore.save_local(index_path)
                    print('Indice salvo em {}'.format(index_path))
    
                self._setup_retriever()
                self._create_rag_chain()
                self.documents_loaded = True
    
                print('Sistema RAG ativo.')
                return True
    
            except Exception as e:
                print('Erro: {}'.format( str(e) ))
                return False            

    def _process_pdfs(self, pdf_paths:List[str]) -> List[Document]:
        """Processa pdfs e retorna chunks"""
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
        )

        for pdf in pdf_paths:
            if not os.path.exists(pdf):
                print('arquivo não encontrado: {}'.format(pdf))
                continue

            print('Processando: {}'.format( Path(pdf).name ))
            loader = PyMuPDFLoader(pdf)
            docs = loader.load()

            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print('{} paginas -> {} chunks'.format( len(docs), len(chunks) ))

        return all_chunks

    def _load_vectorstore(self, path:str):
        """Carrega índice FAISS do disco"""
        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _setup_retriever(self):
        """Configura retriever com MMR"""
        self.retriever = self.vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={
                'k': self.config["retriever_k"],
                "fetch_k": self.config["retriever_fetch_k"],
                "lambda_mult": 0.7
            }
        )

    def _create_rag_chain(self):
        """Cria chain RAG com contextualização de histórico"""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Dado o histórico de conversação e a pergunta atual do usuário,
                          reformule a pergunta para que seja independente do contexto anterior.
                          Mantenha o idioma original (português).
                          NÃO responda a pergunta, apenas reformule-a se necessário."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        contextualize_chain = contextualize_q_prompt | self.llm | StrOutputParser()

        def contextualize_or_not(x: dict) -> str:
            """Reformula se houver histórico, senão usa pergunta original."""
            if x.get("chat_history") and len(x["chat_history"]) > 1: # tem historico com 2 msgs?
                return contextualize_chain.invoke(x)
            return x["input"]

        # retriever contextualizado
        self.contextualized_retriever = (
            RunnablePassthrough.assign(
                reformulated_question=lambda x: contextualize_or_not(x)
            )
            | (lambda x: self.retriever.invoke(x["reformulated_question"]))
        )

        # chain RAG completa
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente especializado em responder perguntas baseadas em documentos.

                            INSTRUÇÕES:
                            - Use APENAS o contexto fornecido abaixo para responder
                            - Se não souber a resposta, diga "Não encontrei essa informação nos documentos"
                            - Cite a fonte (página) quando possível
                            - Seja conciso e objetivo
                            - Responda em português
                            
                            Contexto:
                            {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def format_docs(docs: List[Document]) -> str:
            """Formata documentos para string."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                source = Path(doc.metadata.get("source", "")).name
                formatted.append(f"[Doc {i} | Pág. {page} | {source}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        # chain
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(
                    self.contextualized_retriever.invoke(x)
                )
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(
        self, 
        user_input: str, 
        use_rag: bool = True, 
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Processa pergunta do usuário.

        Args:
            user_input: Pergunta
            use_rag: Se True, usa documentos; se False, chat puro
            verbose: Se True, mostra detalhes

        Returns:
            Dict com answer, sources (se RAG), time, etc.
        """
        start_time = time.time()

        if verbose:
            print(f"\n👤 Você: {user_input}")

        # modo RAG
        if use_rag and self.documents_loaded:
            if verbose:
                print("🔍 Buscando contexto...")
            
            response = self.rag_chain.invoke({
                "input": user_input,
                "chat_history": self.history
            })

            # sources para metadados
            sources = self.contextualized_retriever.invoke({
                "input": user_input,
                "chat_history": self.history
            })

            # apendar historico
            self.history.append(HumanMessage(content=user_input))
            self.history.append(AIMessage(content=response))

            elapsed = time.time() - start_time

            if verbose:
                print(f"\n🤖 Assistente: {response}")
                print(f"\n📚 Fontes consultadas: {len(sources)}")
                for i, doc in enumerate(sources[:3], 1):  # top 3
                    page = doc.metadata.get("page", "?")
                    source = Path(doc.metadata.get("source", "")).name
                    print(f"   [{i}] {source} - pág. {page}")
                print(f"\n⏱️  Tempo: {elapsed:.2f}s")

            # limit histórico
            self._trim_history()

            return {
                "answer": response,
                "sources": sources,
                "time": elapsed,
                "mode": "RAG"
            }

        # modo chat
        else:
            if use_rag and not self.documents_loaded:
                if verbose:
                    print("RAG solicitado mas documentos não carregados. Usando modo chat.")

            # gerar e apendar
            self.history.append(HumanMessage(content=user_input))
            response = self.llm.invoke(self.history)
            self.history.append(AIMessage(content=response.content))

            elapsed = time.time() - start_time

            if verbose:
                print(f"\n🤖 Assistente: {response.content}")
                print(f"\n⏱️  Tempo: {elapsed:.2f}s (modo chat)")

            self._trim_history()

            return {
                "answer": response.content,
                "sources": [],
                "time": elapsed,
                "mode": "CHAT"
            }

    def _trim_history(self):
        """Limita tamanho do histórico mantendo system message."""
        max_msgs = self.config["max_history_messages"] * 2  # Human + AI
        if len(self.history) > max_msgs + 1:  # +1 para system message
            # manter system message e últimas mensagens
            self.history = [self.history[0]] + self.history[-max_msgs:]


    def clear_history(self):
        """Limpa histórico mantendo system message."""
        system_msg = self.history[0] if self.history else None
        self.history = [system_msg] if system_msg else []
        print("Histórico limpo!")

    def clear_documents(self):
        """Remove documentos carregados."""
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.contextualized_retriever = None
        self.documents_loaded = False
        print("Documentos removidos!")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema."""
        stats = {
            "documents_loaded": self.documents_loaded,
            "history_messages": len(self.history),
            "model": self.config["model"],
            "embeddings": self.config["embeddings_model"],
        }

        if self.vectorstore:
            stats["vectors_in_index"] = self.vectorstore.index.ntotal

        return stats

    def save_index(self, path: str = None):
        """Salva índice FAISS manualmente."""
        if not self.vectorstore:
            print("Nenhum índice para salvar!")
            return

        path = path or self.config["index_path"]
        self.vectorstore.save_local(path)
        print(f"Índice salvo em: {path}")
