import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF
import io

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)  # Reset the file pointer to the beginning
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """You are a virtual Research Assistant. Your task is to answer questions related to research papers, 
    including details such as the title, abstract, keywords, field of research, and summary. Provide a 
    thorough and accurate response based on the provided context. Answer the question as detailed as 
    possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])


    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)


    question = user_question.encode("latin-1", errors="replace").decode("latin-1")
    answer = response["output_text"].encode("latin-1", errors="replace").decode("latin-1")
    
    pdf.multi_cell(0, 10, f"Question: {question}\n\nAnswer:\n{answer}")


    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))  
    pdf_output.seek(0)

    st.download_button(label="Download Response as PDF", data=pdf_output, file_name="response.pdf", mime="application/pdf")



def main():
    st.set_page_config("Virtual RA")
    st.header("Virtual Research Assistant")

    user_question = st.text_input("Ask Question from the uploaded Research Paper")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Virtual RA")
        pdf_docs = st.file_uploader("Upload Research Paper and Click on the Submit Button", accept_multiple_files=False)
        if st.button("Submit"):
            with st.spinner("⏳ Please Wait... ⏳"):
                raw_text = get_pdf_text([pdf_docs])
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done! You Can Ask Now!🎉")

if __name__ == "__main__":
    main()
