
import gradio as gr
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. LOAD MODELS
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading language model...")

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

print("Models loaded successfully.")


# ============================================================
# 2. GLOBAL DOCUMENT STATE
# ============================================================

current_chunks = []
current_index = None


# ============================================================
# 3. EXTRACT TEXT FROM PDF
# ============================================================

def process_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# ============================================================
# 4. CREATE TEXT CHUNKS
# ============================================================

def create_chunks(
    pages,
    chunk_size=700,
    overlap=100
):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            chunk_text = text[start:end]

            chunks.append(
                {
                    "text": chunk_text,
                    "page": page["page"]
                }
            )

            if end == len(text):
                break

            start = end - overlap

    return chunks


# ============================================================
# 5. BUILD FAISS VECTOR INDEX
# ============================================================

def build_index(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False
    )

    embedding_array = np.array(
        embeddings
    ).astype("float32")

    dimension = embedding_array.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embedding_array
    )

    return index


# ============================================================
# 6. LOAD AND PROCESS DOCUMENT
# ============================================================

def load_document(pdf_path):

    global current_chunks
    global current_index

    if pdf_path is None:

        return "Please upload a PDF first."

    try:

        pages = process_pdf(pdf_path)

        if not pages:

            return (
                "No readable text was found "
                "in this PDF."
            )

        current_chunks = create_chunks(
            pages
        )

        current_index = build_index(
            current_chunks
        )

        return (
            "Document processed successfully.\n\n"
            f"Pages: {len(pages)}\n"
            f"Chunks: {len(current_chunks)}"
        )

    except Exception as error:

        return (
            "Error processing document:\n"
            f"{str(error)}"
        )


# ============================================================
# 7. ANSWER QUESTIONS
# ============================================================

def answer_question(question):

    global current_chunks
    global current_index

    if current_index is None:

        return (
            "Please upload and process "
            "a PDF first.",
            ""
        )

    if not question or not question.strip():

        return (
            "Please enter a question.",
            ""
        )

    try:

        # ----------------------------------------------------
        # Convert question into embedding
        # ----------------------------------------------------

        question_embedding = (
            embedding_model.encode(
                [question]
            )
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")


        # ----------------------------------------------------
        # Retrieve most relevant chunks
        # ----------------------------------------------------

        top_k = min(
            3,
            len(current_chunks)
        )

        distances, indices = (
            current_index.search(
                question_embedding,
                top_k
            )
        )


        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context_parts = []

        for i in indices[0]:

            context_parts.append(
                f"Page {current_chunks[i]['page']}:\n"
                f"{current_chunks[i]['text']}"
            )

        context = "\n\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # Build prompt
        # ----------------------------------------------------

        prompt = f"""
You are an engineering research assistant.

Answer the question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say:

"I could not find this information in the provided document."

Do not invent information.
Do not use outside knowledge.

Context:

{context}

Question:

{question}

Answer:
"""


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=True
        )

        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )


        # ----------------------------------------------------
        # Remove prompt from generated response
        # ----------------------------------------------------

        if "Answer:" in response:

            response = response.split(
                "Answer:",
                1
            )[1].strip()


        # ----------------------------------------------------
        # Get source pages
        # ----------------------------------------------------

        sources = []

        for i in indices[0]:

            page = current_chunks[i]["page"]

            if page not in sources:

                sources.append(page)


        source_text = ", ".join(
            f"Page {page}"
            for page in sources
        )

        return response, source_text


    except Exception as error:

        return (
            f"Error answering question:\n{str(error)}",
            ""
        )


# ============================================================
# 8. GRADIO USER INTERFACE
# ============================================================

with gr.Blocks() as app:

    gr.Markdown(
        "# Engineering Research Assistant"
    )

    gr.Markdown(
        "Upload an engineering research paper "
        "and ask questions about it using "
        "Retrieval-Augmented Generation (RAG)."
    )


    # --------------------------------------------------------
    # PDF Upload
    # --------------------------------------------------------

    pdf_upload = gr.File(
        label="Upload Engineering Research Paper",
        file_types=[".pdf"],
        type="filepath"
    )


    # --------------------------------------------------------
    # Process Button
    # --------------------------------------------------------

    process_button = gr.Button(
        "Process PDF"
    )


    # --------------------------------------------------------
    # Document Status
    # --------------------------------------------------------

    status_box = gr.Textbox(
        label="Document Status",
        lines=3
    )


    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question_box = gr.Textbox(
        label="Ask a Question",
        placeholder=(
            "e.g. What is the compressive strength?"
        ),
        lines=2
    )


    # --------------------------------------------------------
    # Ask Button
    # --------------------------------------------------------

    ask_button = gr.Button(
        "Ask"
    )


    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer_box = gr.Textbox(
        label="Answer",
        lines=6
    )


    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources_box = gr.Textbox(
        label="Source Pages",
        lines=2
    )


    # --------------------------------------------------------
    # Button Actions
    # --------------------------------------------------------

    process_button.click(
        fn=load_document,
        inputs=pdf_upload,
        outputs=status_box
    )


    ask_button.click(
        fn=answer_question,
        inputs=question_box,
        outputs=[
            answer_box,
            sources_box
        ]
    )


# ============================================================
# 9. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":

    app.launch()
