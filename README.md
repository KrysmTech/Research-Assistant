**Engineering Research Assistant — RAG-Based Document Question Answering**

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, ask natural-language questions, and receive answers grounded in retrieved passages from the document, together with source-page references.

**Project overview**

The project explores how semantic search, vector retrieval, and a local language model can be combined into a complete document-intelligence application.

The initial demonstration uses an engineering research paper on SCABA-lime concrete, but the underlying pipeline is domain-agnostic and can be adapted to other document types.

**Problem**

Technical and research documents can contain large amounts of information. Finding a specific method, result, definition, or value often requires manually searching through many pages.

The goal was to build a system that lets a user ask questions in natural language and retrieve relevant information from the uploaded document.

**Solution**

The application implements a RAG pipeline:

PDF text extraction with PyPDF

Page-aware text chunking with overlap

Sentence-transformer embeddings

FAISS vector similarity search

Retrieval of the most relevant document chunks

Context-grounded generation with a local instruction-tuned language model

Source-page attribution

Gradio user interface

**Architecture**

User
  |
  v
Upload PDF
  |
  v
PDF Text Extraction (PyPDF)
  |
  v
Page-aware Chunking
  |
  v
Sentence Transformer Embeddings
  |
  v
FAISS Vector Index
  |
  +----------------------+
  |                      |
  v                      v
User Question       Indexed Chunks
  |                      |
  v                      |
Question Embedding       |
  |                      |
  +----------+-----------+
             |
             v
      Semantic Retrieval
             |
             v
     Relevant PDF Chunks
             |
             v
      Local LLM (SmolLM2)
             |
             v
   Grounded Answer + Sources
             |
             v
        Gradio UI

**Technical implementation**

**Document processing**

PDF pages are extracted while preserving page numbers. Text is split into overlapping chunks so that relevant context is less likely to be lost at chunk boundaries.

**Semantic retrieval**

Each chunk is transformed into a 384-dimensional embedding using sentence-transformers/all-MiniLM-L6-v2.

FAISS is then used to index the embeddings and retrieve the nearest chunks for a user's question.

**Generation**

The retrieved chunks are inserted into a constrained prompt for a local instruction-tuned model (HuggingFaceTB/SmolLM2-135M-Instruct).

The prompt instructs the model to answer only from the supplied context and to state when the information cannot be found.

**Source attribution**

The original page number is stored with every chunk. After retrieval, the application returns the pages associated with the retrieved context.

**Current result**

The working prototype successfully:

extracts text from a PDF;

creates 38 chunks from the demonstration document;

generates 384-dimensional embeddings;

indexes all chunks in FAISS;

retrieves relevant passages for natural-language questions;

generates context-grounded answers with a local model;

returns source pages;

exposes the pipeline through a Gradio interface.

**Technology stack**

Python

PyPDF

Sentence Transformers

FAISS

NumPy

Hugging Face Transformers

PyTorch

Gradio

Google Colab for development

**Why this project matters**

This project demonstrates practical ML engineering beyond simply calling an LLM API. It combines:

NLP preprocessing

embeddings

vector search

information retrieval

prompt design

local model inference

application integration

source attribution

It also provides a foundation for document search systems in research, education, engineering, legal, finance, and enterprise knowledge management.

**Limitations**

The current prototype is intentionally lightweight; I made it so.

The local language model is small and has limited generation quality compared with larger models.

Scanned/image-only PDFs require OCR, which is not yet implemented.

Retrieval quality has not yet been evaluated systematically against a labelled question set.

The prototype is not yet optimized for concurrent users.

Colab is being used for development rather than permanent hosting.

Next improvements

Add retrieval evaluation using a labelled question set.

Add OCR support for scanned PDFs.

Add document-level metadata and better chunking.

Compare retrieval strategies and embedding models.

Add answer-confidence and citation validation.

Deploy the application to a persistent hosting platform.

Add multi-document collections.

Add authentication and usage controls if deployed publicly.

**Portfolio positioning**

Project title: Engineering Research Assistant — RAG-Based Document Question Answering

Short description:
Built an end-to-end Retrieval-Augmented Generation system that uses semantic embeddings and FAISS vector search to retrieve relevant passages from technical PDFs and generate context-grounded answers with source-page references.

Core skills demonstrated:
Machine Learning, NLP, RAG, semantic search, embeddings, vector databases, local LLM inference, Python, model integration, evaluation planning.

**Links**

Live demo: Add after deployment

Source code: Add GitHub repository

Demo video: Add after recording
