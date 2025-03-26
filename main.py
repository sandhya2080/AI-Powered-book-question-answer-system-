import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import logging
import traceback
import numpy as np
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model_qa = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# Extracting text from pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    logger.debug(f"Extracted text from page: {page_text[:100]}")  # Print first 100 characters for debug
                    text += page_text
                else:
                    logger.warning("Warning: No text extracted from page")
            except Exception as e:
                logger.error(f"Error extracting text from page: {e}")
                logger.error(traceback.format_exc())
    return text

#Cleaning the text
def clean_text(text):
    try:
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-printable characters
        logger.debug(f"Cleaned text: {text[:100]}")  # Print first 100 characters of cleaned text
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        logger.error(traceback.format_exc())
        return text

#Chunking
def get_text_chunks(text):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        logger.debug(f"Number of chunks created: {len(chunks)}")
        logger.debug(f"First chunk: {chunks[0][:100]}")  # Print first 100 characters of the first chunk
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        logger.error(traceback.format_exc())
        return []

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#Embedding
def get_vectorstore(text_chunks):
    try:
        logger.debug(f"Creating embeddings for text chunks: {text_chunks[:3]}...")  # Print first 3 chunks for debug
        embeddings = embedding_model.encode(text_chunks)
        docstore = {i: {'text': text, 'embedding': embedding} for i, (text, embedding) in enumerate(zip(text_chunks, embeddings))}
        logger.debug(f"Vectorstore created with {len(docstore)} entries")
        return docstore
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        logger.error(traceback.format_exc())
        return None

#Conversation Chain
def get_conversation_chain(vectorstore):
    try:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        def similarity_search(query, k=5):
            query_embedding = embedding_model.encode([query])[0]
            similarities = [
                (i, np.dot(query_embedding, entry['embedding']) / (np.linalg.norm(query_embedding) * np.linalg.norm(entry['embedding'])))
                for i, entry in vectorstore.items()
            ]
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            return [vectorstore[i]['text'] for i, _ in similarities[:k]]

        def conversation_function(question):
            relevant_chunks = similarity_search(question['question'], k=5)
            context = "\n".join([chunk for chunk in relevant_chunks])
            result = qa_pipeline(question=question['question'], context=context)
            #logger.debug(f"QA Pipeline result: {result}")  # Log the QA result
            answer = result['answer']
            return {'chat_history': [{'content': f"Answer to {question['question']}: <br><br>{answer}"}]}

        logger.debug("Conversation chain created")
        return conversation_function
    except Exception as e:
        logger.error(f"Error creating conversation chain: {e}")
        logger.error(traceback.format_exc())
        return None


def initialize_conversation():
    def conversation_function(question):
        return {'chat_history': [{'content': f"Answer to {question['question']}"}]}
    return conversation_function

if 'conversation' not in st.session_state:
    st.session_state.conversation = initialize_conversation()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

#User Input
def handle_userinput(user_question):
    logger.debug(f"st.session_state.conversation: {st.session_state.conversation}")
    logger.debug(f"type: {type(st.session_state.conversation)}")

    if st.session_state.conversation is None:
        st.error("Conversation object is not initialized.")
        return "Error: Conversation object is not initialized."

    response = st.session_state.conversation({'question': user_question})
    #logger.debug(f"Response from conversation function: {response}")  # Log the response
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

#PDF Processing
def process_pdf(pdf_docs):
    try:
        raw_text = get_pdf_text(pdf_docs)
        if not raw_text:
            st.error("No text could be extracted from the uploaded PDFs.")
            return
        cleaned_text = clean_text(raw_text)
        text_chunks = get_text_chunks(cleaned_text)
        if not text_chunks:
            st.error("Text splitting failed. No chunks created.")
            return
        vectorstore = get_vectorstore(text_chunks)
        if not vectorstore or len(vectorstore) == 0:
            st.error("Vectorstore creation failed or returned empty embeddings.")
            return
        st.session_state.conversation = get_conversation_chain(vectorstore)
        if not st.session_state.conversation:
            st.error("Conversation chain creation failed.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())

#Main func
def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Question Answering", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF Question Answering :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                process_pdf(pdf_docs)

if __name__ == '__main__':
    main()
