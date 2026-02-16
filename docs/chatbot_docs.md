# Documentação do Módulo chatbot.py

## Visão Geral
O Módulo `chatbot.py` implementa a lógica de backend do sistema RAG (Retrieval Augmented Generation) para conversação com documentos PDFs.

**Autor:** Wellington M. Santos

**Data:** Fevereiro/2026

---

## Arquitetura

### Diagrama de Componentes
![image](../assets/arquitetura_chatbot_diagrama.png)

### Princípios de Design
1. **Lazy Loading**: Índices são carregados do disco quando existentes, criados apenas quando necessário.
2. **Configuração Dinâmica**: Parâmetros podem ser alterados em runtime com recriação seletiva de componentes.
3. **Persistência Automática**: Índices FAISS são salvos automaticamente em disco.
4. **Modo RAG Estrito**: Respostas baseadas exclusivamente no documentos fornecidos.
5. **Busca Híbrida**: Combina resultados de múltiplos índices com ranking global.

---

## Dependências

### Bibliotecas Core
```python
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
```

### Langchain Ecosystem
```python
# LLM provider
from langchain_groq import ChatGroq

# core componentes
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthough

# document processing & storage
from lanchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS

# embeddings & text processing
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---
