# importing dependencies
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os, glob
from pathlib import Path

# Import openai and google_genai as main LLM services
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

from langchain.schema import format_document
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)

# text_splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.document_loaders import PDFPlumberLoader
# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Contextual_compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Cohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere

# HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# Import streamlit
import streamlit as st

# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
DATA_PATH = '/Users/chinmay/Downloads/Acad/Projects/TAMU-CS-RAG/Data/CS_subjects'
st.session_state.temperature = 0.7
st.session_state.top_p = 0.9


def get_chunks_from_pdfs_in_directory(directory_path):
    """
    Get text chunks from PDF files in a directory using PyPDFDirectoryLoader.
    """
    # Instantiate the PyPDFDirectoryLoader with the directory path
    pdf_loader = PyPDFDirectoryLoader(directory_path, )
    
    # Load and split PDF documents
    documents = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],    
        chunk_size = 2000,
        chunk_overlap= 200)
    )
    
    return documents

# extracting text from pdf
def get_pdf_text(docs):
    text=""
    for pdf in docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# converting text to chunks
def get_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    return chunks


# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vectorstore=faiss.FAISS.from_texts(texts=chunks,embedding=embeddings)
    
    return vectorstore


# generating conversation chain  
def get_conversationchain(vectorstore):
    # llm=ChatOpenAI(temperature=0.2)
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        # model="gemini-pro",
        model="gemini-pro",
        temperature=0.5,
        top_p=0.95,
        convert_system_message_to_human=True
        )
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True,
                                      output_key='answer') # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=vectorstore.as_retriever(),
                                condense_question_prompt=CUSTOM_QUESTION_PROMPT,
                                memory=memory)
    return conversation_chain


# generating response from user queries and displaying them accordingly
def handle_question(question):
    response=st.session_state.conversation({'question': question})
    st.session_state.chat_history=response["chat_history"]
    for i,msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg.content,),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)


def Vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever

def create_memory(memory_max_token=None):
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        input_key="question",
    )
    return memory


def create_compression_retriever(embeddings, base_retriever, chunk_size=1000, k=16, similarity_threshold=None):
    # 1. splitting docs into smaller chunks
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separator=". ")
    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query
    relevant_filter = EmbeddingsFilter(
        embeddings=embeddings, k=k, similarity_threshold=similarity_threshold )

    # 4. Reorder the documents
    reordering = LongContextReorder()

    # 5. create compressor pipeline and retriever
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering])
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=base_retriever)
    return compression_retriever

def create_retriever(
    vector_store,
    embeddings,
    base_retriever_search_type="similarity",  # Specify the default search type for the base retriever
    base_retriever_k=16,  # Specify the default value for 'k' (number of retrievals) for the base retriever
    compression_retriever_k=20,  # Specify the default value for 'k' for the compression retriever
):
    # Create the base retriever using the provided vector store and settings
    base_retriever = Vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,  
    )

    # Create the compression retriever using embeddings and the base retriever
    compression_retriever = create_compression_retriever(
        embeddings=embeddings,
        base_retriever=base_retriever,
        k=compression_retriever_k,
    )

    return compression_retriever  


def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context
    to the `LLM` wihch will answer."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template

def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="english",
):


    # 1. Define the standalone_question prompt.
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question, in its original language.\n\n
        Chat History:\n{chat_history}\n
        Follow Up Input: {question}\n
        Standalone question:""",
    )

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents)
    # to the `LLM` wihch will answer

    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    memory = create_memory()

    # 4. Instantiate LLMs: standalone_query_generation_llm & response_generation_llm
    standalone_query_generation_llm = ChatGoogleGenerativeAI(
        google_api_key=st.session_state.google_api_key,
        model="gemini-pro",
        temperature=0.1,
        convert_system_message_to_human=True,
    )

    # response generation
    response_generation_llm = ChatGoogleGenerativeAI(
        google_api_key=st.session_state.google_api_key,
        model="gemini-pro",
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        convert_system_message_to_human=True,
    )

    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory

def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        
        # 1. Invoke LLM
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        # 2. Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant", avatar='/Users/chinmay/Downloads/Acad/Projects/TAMU-CS-RAG/tamu_logo.png'):
            # 2.1. Display anwser:
            st.markdown(answer)

            # 2.2. Display source documents:
            with st.expander("**Source documents**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Page: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Source: "
                        + str(document.metadata["source"])
                        + page
                        + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"
                st.markdown(prompt)                
                st.markdown(documents_content)

    except Exception as e:
        st.warning(e)

def main():
    
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Texas A&M Computer Science Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Hi, how can I help you?",
            }
        ]
    for msg in st.session_state.messages:
        if msg["role"] == 'assistant':
            st.chat_message(msg["role"], avatar='/Users/chinmay/Downloads/Acad/Projects/TAMU-CS-RAG/tamu_logo.png').write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():
        if not st.session_state.chain:
            st.info(
                f"Please add docs to continue."
            )
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)

    
    st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY")
    
    with st.sidebar:
        st.subheader("Your documents")
        # docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                
                #get the text chunks
                text_chunks = get_chunks_from_pdfs_in_directory(DATA_PATH)

                #creating embedding from Google Gemini
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001", google_api_key=st.session_state.google_api_key
                )
                st.session_state.vector_store = Chroma.from_documents(
                    documents=text_chunks,
                    embedding=embeddings,
                    persist_directory= '/Users/chinmay/Downloads/Acad/Projects/TAMU-CS-RAG/RAG pipeline/Chroma',
                )                

                #create contextual compresed vectorstore
                st.session_state.retriever = create_retriever(
                    vector_store=st.session_state.vector_store,
                    embeddings=embeddings,
                    base_retriever_search_type="similarity",
                    base_retriever_k=20,
                    compression_retriever_k=15,
                )
                
                (st.session_state.chain, st.session_state.memory) = create_ConversationalRetrievalChain(
                    retriever=st.session_state.retriever,
                    chain_type="stuff",
                )
            st.success('Documents uploaded and processed.')
    
if __name__ == '__main__':
    main()