import streamlit as st
import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Engineering Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Engineering Research Assistant")

st.write(
    "Upload a PDF document, ask questions in natural "
    "language, and explore answers grounded in the "
    "document with source-page references."
)


# =====================================================
# LOAD MODELS
# Models are cached to avoid reloading on every rerun.
# =====================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_language_model():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    model.eval()

    return tokenizer, model


# =====================================================
# EXTRACT TEXT FROM PDF
# =====================================================

def extract_pages(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []

    for page_number, page in enumerate(
        reader.pages, start=1
    ):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


# =====================================================
# SPLIT TEXT INTO CHUNKS
# =====================================================

def create_chunks(pages, chunk_size=700, overlap=100):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append({
                "text": text[start:end],
                "page": page["page"]
            })

            if end == len(text):
                break

            start = end - overlap

    return chunks


# =====================================================
# CREATE FAISS SEARCH INDEX
# =====================================================

def build_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=8,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings, dtype=np.float32
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index


# =====================================================
# ANSWER QUESTIONS USING RETRIEVAL-AUGMENTED
# GENERATION (RAG)
# =====================================================

def answer_question(
    question,
    chunks,
    index,
    embedding_model,
    tokenizer,
    model
):
    if not question.strip():
        return "Please enter a question.", []

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.asarray(
        question_embedding, dtype=np.float32
    )

    top_k = min(3, len(chunks))

    distances, indices = index.search(
        question_embedding, top_k
    )

    context_parts = []
    source_pages = []

    for chunk_index in indices[0]:
        if chunk_index < 0:
            continue

        chunk = chunks[chunk_index]

        context_parts.append(
            f"Page {chunk['page']}:\n{chunk['text']}"
        )

        if chunk["page"] not in source_pages:
            source_pages.append(chunk["page"])

    context = "\n\n".join(context_parts)

    prompt = f"""You are a document research assistant.

Answer the question using only the information
provided in the context.

If the answer cannot be found in the context,
say: "I could not find this information in the
provided document."

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated answer tokens.
    input_length = inputs["input_ids"].shape[1]

    new_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    if not answer:
        answer = (
            "The model did not generate an answer. "
            "Try asking the question differently."
        )

    return answer, source_pages


# =====================================================
# DOCUMENT UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    if st.button("Process PDF", type="primary"):

        try:
            with st.spinner(
                "Loading models and processing your PDF..."
            ):
                embedding_model = load_embedding_model()

                tokenizer, model = load_language_model()

                pages = extract_pages(uploaded_file)

                if not pages:
                    st.error(
                        "No readable text was found. "
                        "The PDF may be scanned or image-only."
                    )
                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:
                    st.error(
                        "No text chunks could be created."
                    )
                    st.stop()

                index = build_index(
                    chunks, embedding_model
                )

                # Save the processed document for this session.
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["filename"] = (
                    uploaded_file.name
                )

            st.success(
                f"Processed {uploaded_file.name} successfully!"
            )

            st.write(f"Pages with text: {len(pages)}")
            st.write(f"Text chunks: {len(chunks)}")

        except Exception as error:
            st.error(
                f"Could not process the PDF: {error}"
            )


# =====================================================
# QUESTION-ANSWERING INTERFACE
# =====================================================

if "chunks" in st.session_state:
    st.divider()

    st.subheader("Ask questions about your document")

    st.caption(
        f"Current document: {st.session_state['filename']}"
    )

    question = st.text_input(
        "Your question",
        placeholder=(
            "What are the main findings of this document?"
        )
    )

    if st.button("Get Answer"):
        try:
            with st.spinner(
                "Searching the document and generating an answer..."
            ):
                embedding_model = load_embedding_model()
                tokenizer, model = load_language_model()

                answer, source_pages = answer_question(
                    question,
                    st.session_state["chunks"],
                    st.session_state["index"],
                    embedding_model,
                    tokenizer,
                    model
                )

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Source pages")

            if source_pages:
                st.write(
                    ", ".join(
                        f"Page {page}"
                        for page in source_pages
                    )
                )
            else:
                st.write("No source pages available.")

        except Exception as error:
            st.error(
                f"Could not answer the question: {error}"
            )


st.divider()

st.caption(
    "Built with Python, Streamlit, Sentence Transformers, "
    "FAISS, PyTorch and SmolLM2."
)