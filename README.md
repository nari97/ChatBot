# WhatsApp Chat Analyzer and Query Engine

## Overview

This project implements a sophisticated chat analysis and query system using graph database technologies, retrieval augmented generation (RAG), and large language models (LLMs). It processes WhatsApp chat data, creates a structured database, generates conversation summaries, and provides a question-answering interface based on the chat content.

## Key Technologies

- Database: Neo4j (Graph Database)
- Language Model: Mistral (via LangChain)
- Embeddings: HuggingFace's all-MiniLM-L6-v2
- Vector Database: Faiss
- RAG Framework: LangChain
- Data Processing: Pandas, NumPy
- Python 3.x

## System Architecture

### 1. Data Ingestion and Database Creation

- **File**: `chat_neo4j_etl/ingest_basic_schema.py`
- **Process**: 
  - Parses WhatsApp chat export file
  - Creates a graph structure in Neo4j
  - Nodes: Users, Messages
  - Relationships: SENT (User to Message)
  - Properties: Timestamp (day, month, year, hours, minutes)

### 2. Conversation Summarization

- **File**: `chat_neo4j_etl/summarize_conversations.py`
- **Process**:
  - Randomly selects days from the database
  - Chunks messages (30 messages per chunk)
  - Utilizes LangChain with Mistral model for summarization
  - Agent: `agents/conversation_summarizer_agent.py`
  - Creates Conversation nodes with summary and sentiment analysis

### 3. ChatBot (In Development)

#### 3.1 Vector Database Creation

- Uses Faiss for efficient similarity search
- Embeddings: HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
- Stores embeddings of conversation summaries

#### 3.2 Query Processing Pipeline

a) **Timeframe Extraction**
   - Agent: `agents/timeframe_detection_agent.py`
   - Extracts relevant time period from user query

b) **Context Retrieval**
   - Agent: `agents/query_retriever_agent.py`
   - Uses RAG techniques to retrieve top 10 relevant conversation summaries
   - Filters based on extracted timeframe

c) **Response Generation and Verification**
   - Agent: `agents/citation_verification_agent.py`
   - Generates response based on retrieved context
   - Adds citations to original conversation summaries

## Current Status and Future Work

- The ChatBot component is currently under active development
- Future plans include:
  - Improving retrieval accuracy
  - Enhancing the user interface
  - Implementing additional analysis features

