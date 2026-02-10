# -*- coding: utf-8 -*-
"""
RAGChatBot 
Combina: Persistência automática + Contextualização LCEL + Modo híbrido

Recursos:
- Persistência FAISS automática (índice salvo em disco)
- Contextualização de perguntas com histórico (LCEL)
- Múltiplos PDFs 
"""

################################################################################
# IMPORTS
################################################################################

import os
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List, Union
from datetime import datetime
import json
import logging
import unicodedata
import re

# LLM 
from langchain_groq import ChatGroq

## core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

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
    "index_dir": "indices",
}

LLM_DEPENDENT = ['model', 'temperature', 'max_tokens']
RETRIEVER_DEPENDENT = ['retriever_k', 'retriever_fetch_k']
EMBEDDING_DEPENDENT = ['embeddings_model', 'chunk_size', 'chunk_overlap']

################################################################################
# CLASSE PRINCIPAL
################################################################################

class ChatBot:
    """
    ChatBot com Lazy Loading e Configuração Dinâmica.
    """

    def __init__(self, config: Dict[str, Any] = None, api_key: str = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.api_key = api_key or os.getenv('GROQ_API_KEY')

        if not self.api_key:
            raise ValueError('GROQ_API_KEY não configurada!')

        # componentes
        self.llm = None
        self.embeddings = None
        self.history: List[Any] = []

        # indices
        self.indices: Dict[str, FAISS] = {}
        self.retrievers: Dict[str, Any] = {}

        logger.info('\nInicializando ChatBot...')
        self._init_llm()
        self._init_embeddings()
        self._init_system_prompt()
        logger.info('ChatBot pronto!\n')

    def _init_llm(self):
        """Inicializa LLM."""
        self.llm = ChatGroq(
            model=self.config['model'],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens'],
            api_key=self.api_key
        )
        logger.info(f'LLM: {self.config["model"]} (temp={self.config["temperature"]}, tokens={self.config["max_tokens"]})')

    def _init_embeddings(self):
        """Inicializa embeddings."""
        logger.info(f"Embeddings: {self.config['embeddings_model']}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings_model'],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

    def _init_system_prompt(self):
        """System prompt."""
        system_msg = SystemMessage(
            content="""Você é um assistente especializado em responder com base em documentos fornecidos.""")
        self.history.append(system_msg)

    # ========================================================================
    # UPDATE CONFIG -> PARA FRONTEND
    # ========================================================================

    def update_config(self, **kwargs) -> Dict[str, Any]:
        """
        Atualiza configurações dinamicamente. Recria componentes afetados.

        Args:
            **kwargs: Configs a atualizar (model, temperature, max_tokens, 
                     retriever_k, retriever_fetch_k, etc.)

        Returns:
            Dict com success, message, changes, recreated, current_config
        """
        result = {
            'success': False,
            'message': '',
            'changes': [],
            'recreated': [],
            'current_config': {},
            'warning': None
        }

        # validar chaves
        valid_keys = set(DEFAULT_CONFIG.keys())
        invalid = [k for k in kwargs if k not in valid_keys]
        if invalid:
            result['message'] = f"Inválidas: {', '.join(invalid)}"
            result['current_config'] = self.get_config()
            return result

        # validar valores
        errors = []
        if 'temperature' in kwargs:
            t = kwargs['temperature']
            if not isinstance(t, (int, float)) or not (0.0 <= t <= 2.0):
                errors.append("temperature: 0.0-2.0")

        if 'max_tokens' in kwargs:
            if not isinstance(kwargs['max_tokens'], int) or kwargs['max_tokens'] < 1:
                errors.append("max_tokens: inteiro > 0")

        if 'retriever_k' in kwargs:
            if not isinstance(kwargs['retriever_k'], int) or kwargs['retriever_k'] < 1:
                errors.append("retriever_k: inteiro > 0")

        if errors:
            result['message'] = "Erros: " + "; ".join(errors)
            result['current_config'] = self.get_config()
            return result

        # detectar mudanças
        changes = []
        needs_llm = False
        needs_retriever = False
        needs_embedding = False

        for key, new_val in kwargs.items():
            old_val = self.config.get(key)
            if old_val != new_val:
                self.config[key] = new_val
                changes.append(f"{key}: {old_val} → {new_val}")

                if key in LLM_DEPENDENT:
                    needs_llm = True
                if key in RETRIEVER_DEPENDENT:
                    needs_retriever = True
                if key in EMBEDDING_DEPENDENT:
                    needs_embedding = True

        if not changes:
            result['success'] = True
            result['message'] = "Nenhuma alteração"
            result['current_config'] = self.get_config()
            return result

        # aplicar mudanças
        try:
            if needs_embedding:
                logger.info("\nRecriando embeddings...")
                self._init_embeddings()
                result['recreated'].append('embeddings')
                # embeddings mudou → índices recriados
                if self.indices:
                    logger.warning("Modelo de embeddings mudou! Índices precisam ser recarregados.")
                    self.indices = {}
                    self.retrievers = {}
                    result['warning'] = "Recarregue os documentos (embeddings alterados)"

            if needs_llm:
                logger.info("Recriando LLM...")
                self._init_llm()
                result['recreated'].append('llm')

            if needs_retriever and self.indices:
                logger.info("Recriando retrievers...")
                self._recreate_retrievers()
                result['recreated'].append('retrievers')

            # recriar chain se LLM ou retriever mudou
            if (needs_llm or needs_retriever) and self.indices:
                logger.info("Recriando RAG chain...")
                self._create_unified_rag_chain()
                result['recreated'].append('rag_chain')

            result['success'] = True
            result['message'] = f"Atualizado ({len(changes)} mudanças)"
            result['changes'] = changes

        except Exception as e:
            logger.exception("Erro ao atualizar config")
            result['message'] = f"Erro: {str(e)}"

        result['current_config'] = self.get_config()
        return result

    def _recreate_retrievers(self):
        """Recria todos os retrievers com novos parâmetros."""
        self.retrievers = {}
        for name, vectorstore in self.indices.items():
            self.retrievers[name] = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.config["retriever_k"],
                    "fetch_k": self.config["retriever_fetch_k"]
                }
            )

    def get_config(self) -> Dict[str, Any]:
        """Retorna config atual."""
        return self.config.copy()

    def reset_config(self) -> Dict[str, Any]:
        """Reseta para padrão."""
        # preservar index_dir e outras configs não-LLM
        preserved = {k: v for k, v in self.config.items() 
                    if k not in DEFAULT_CONFIG or k in ['index_dir']}
        default = DEFAULT_CONFIG.copy()
        default.update(preserved)
        return self.update_config(**default)

    # ========================================================================
    # CARGA PDFs
    # ========================================================================

    def load_documents(self, pdf_paths: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Carrega lista de PDFs com lazy loading.
        Para cada PDF: se índice existe carrega, senão cria.
        """
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        results = {'loaded': [], 'created': [], 'failed': [], 'total': 0}
        base_index_dir = Path(self.config['index_dir'])

        logger.info(f"Processando {len(pdf_paths)} documento(s)...")

        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path).resolve()

            if not pdf_path.exists():
                logger.error(f"Não encontrado: {pdf_path}")
                results['failed'].append({'file': pdf_path, 'error': 'Not found'})
                continue

            pdf_name = pdf_path.stem
            safe_name = self._sanitize_name(pdf_name)
            index_path = base_index_dir / safe_name

            logger.info(f"Processando: {pdf_name}")

            try:
                if index_path.exists() and index_path.is_dir():
                    # CARREGAR DO DISCO
                    logger.info(f"Carregando índice existente...")
                    self._load_single_index(safe_name, index_path)
                    results['loaded'].append(pdf_name)
                    logger.info(f"Carregado")
                else:
                    # CRIAR NOVO
                    logger.info(f"Criando novo índice...")
                    success = self._create_single_index(pdf_path, safe_name, index_path)
                    if success:
                        results['created'].append(pdf_name)
                        logger.info(f"Criado")
                    else:
                        results['failed'].append({'file': pdf_name, 'error': 'Process failed'})

            except Exception as e:
                logger.exception("Erro ao processar PDF %s", pdf_name)
                results['failed'].append({'file': pdf_name, 'error': str(e)})

        results['total'] = len(self.indices)

        logger.info(f"Resumo: {len(results['loaded'])} carregados, {len(results['created'])} criados, {len(results['failed'])} falhas")

        if self.indices:
            self._create_unified_rag_chain()

        return results

    def _sanitize_name(self, name: str) -> str:
        """Nome seguro para pasta."""
        nfkd_form = unicodedata.normalize('NFKD', name) # normalizar unicode
        name_ascii = nfkd_form.encode('ASCII', 'ignore').decode('utf-8') # manter ASCII
        safe = re.sub(r'[^a-zA-Z0-9\-_]', '_', name_ascii) # sub !letras ! num -> '_'
        safe = re.sub(r'_+','_', safe) # duplicados
        return safe[:50]

    def _create_single_index(self, pdf_path: Path, name: str, index_path: Path) -> bool:
        """Cria índice para um PDF usando pathlib."""
        
        # criar diretorio
        try:
            index_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Erro ao criar pasta {index_path}: {e}")
            return False

        # carregar pdf como str
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()

        if not docs:
            return False

        # chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            return False

        for chunk in chunks:
            chunk.metadata['source_file'] = pdf_path.name
            chunk.metadata['source_name'] = name

        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # str tbm para faiss 
        vectorstore.save_local(str(index_path))

        # metadados
        metadata = {
            'original_file': str(pdf_path),
            'name': name,
            'chunks': len(chunks),
            'pages': len(docs),
            'created_at': datetime.now().isoformat(),
            'config': {
                'embeddings_model': self.config['embeddings_model'],
                'chunk_size': self.config['chunk_size']
            }
        }
        
        # json
        with open(index_path / 'info.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.indices[name] = vectorstore
        self.retrievers[name] = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.config["retriever_k"], "fetch_k": self.config["retriever_fetch_k"]}
        )

        return True

    def _load_single_index(self, name: str, index_path: Path):
            """Carrega índice existente."""
            vectorstore = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            self.indices[name] = vectorstore
            self.retrievers[name] = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.config["retriever_k"], "fetch_k": self.config["retriever_fetch_k"]}
            )

    # ========================================================================
    # RAG CHAIN
    # ========================================================================

    def _search_all_indices(self, query: str) -> List[Document]:
        """Busca em TODOS os índices."""
        all_docs = []
        for name, retriever in self.retrievers.items():
            docs = retriever.invoke(query)
            for doc in docs:
                doc.metadata['index_source'] = name
            all_docs.extend(docs)
        return all_docs

    def _create_unified_rag_chain(self):
        """Cria chain que busca em todos os índices."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Reformule considerando histórico. Mantenha português."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        contextualize_chain = contextualize_q_prompt | self.llm | StrOutputParser()

        def get_context(x: dict) -> str:
            if x.get("chat_history") and len(x["chat_history"]) > 1:
                query = contextualize_chain.invoke(x)
            else:
                query = x["input"]

            docs = self._search_all_indices(query)

            formatted = []
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                source = doc.metadata.get("source_file", "?")
                formatted.append(f"[Doc {i} | {source} | Pág. {page}]\n{doc.page_content}")

            return "\n\n---\n\n".join(formatted)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente estritamente baseado em documentos.
Sua tarefa é responder à pergunta do usuário usando APENAS o contexto fornecido abaixo.

Regras Rígidas:
1. NÃO use seu conhecimento prévio ou externo.
2. Se a resposta não estiver explicitamente no contexto, diga: "A informação não consta nos documentos fornecidos."
3. Não invente informações.
4. Cite a fonte (nome do arquivo) sempre que possível no corpo da resposta.

Contexto:
{context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        self.rag_chain = (
            RunnablePassthrough.assign(context=lambda x: get_context(x))
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )

    # ========================================================================
    # CHAT
    # ========================================================================

    def chat(self, user_input: str, verbose: bool = True) -> Dict[str, Any]:
        """Processa pergunta."""
        start_time = time.time()

        if verbose:
            logger.info(f"Entrada do usuário: {user_input}")

        if not self.indices:
            msg = "**Modo RAG Estrito**: Por favor, carregue documentos (PDFs) antes de fazer perguntas."
            if verbose:
                logger.warning('Tentativa de chat sem documentos carregados')

            return {"answer": msg, "mode": "NO_DOCS", "sources": [], "time":0}

        # Modo RAG
        if verbose:
            logger.info(f"Buscando em {len(self.indices)} documento(s)...")

        response = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": self.history
        })

        sources = self._search_all_indices(user_input)

        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=response))

        elapsed = time.time() - start_time

        if verbose:
            logger.info(f"Resposta: {response}")
            logger.info(f"Fontes encontradas: {len(sources)}")
            by_source = {}
            for doc in sources[:6]:
                src = doc.metadata.get('source_file', '?')
                by_source.setdefault(src, []).append(doc.metadata.get('page', '?'))
            for src, pages in by_source.items():
                logger.info(f"{src} (páginas: {', '.join(map(str, pages[:3]))})")
            logger.info(f"Tempo: {elapsed:.2f}s")

        self._trim_history()

        return {"answer": response, "mode": "RAG", "sources": sources, "time": elapsed}

    def _trim_history(self):
        """Limita histórico."""
        max_msgs = self.config["max_history_messages"] * 2
        if len(self.history) > max_msgs + 1:
            self.history = [self.history[0]] + self.history[-max_msgs:]

    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================

    def list_loaded_documents(self) -> List[Dict[str, Any]]:
            docs = []
            base_dir = Path(self.config['index_dir'])
            
            for name in self.indices.keys():
                index_path = base_dir / name
                info_path = index_path / 'info.json'

                info = {}
                if info_path.exists():
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)

                docs.append({
                    'name': name,
                    'original_file': info.get('original_file', '?'),
                    'chunks': info.get('chunks', 0),
                    'pages': info.get('pages', 0),
                    'created': info.get('created_at', '?')
                })
            return docs

    def delete_index(self, name: str) -> bool:
        index_path = Path(self.config['index_dir']) / name

        if not index_path.exists():
            logger.warning(f"Índice '{name}' não existe")
            return False

        if name in self.indices:
            self.remove_document(name)

        shutil.rmtree(index_path)
        logger.info(f"Índice '{name}' deletado do disco")
        return True

    def remove_document(self, name: str) -> bool:
        """Remove documento da memória."""
        if name not in self.indices:
            logger.warning(f"'{name}' não carregado")
            return False

        del self.indices[name]
        del self.retrievers[name]

        if self.indices:
            self._create_unified_rag_chain()
        else:
            self.rag_chain = None

        logger.info(f"'{name}' removido da memória")
        return True

    def clear_all(self):
        """Remove todos da memória."""
        self.indices = {}
        self.retrievers = {}
        self.rag_chain = None
        logger.info("Memória limpa")


    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas."""
        return {
            'documents_in_memory': len(self.indices),
            'document_names': list(self.indices.keys()),
            'history_messages': len(self.history),
            'model': self.config['model'],
            'temperature': self.config['temperature'],
            'embeddings': self.config['embeddings_model']
        }
