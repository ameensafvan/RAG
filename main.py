import os
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.chains import RetrievalQA


#API Key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

app = FastAPI()  #create FastAPI APP

#directory of document fldr
docs_folder = "./docs"
os.makedirs(docs_folder, exist_ok=True)


#here we are using OpenAI Embd Mdl
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Chroma(persist_directory="./chroma_db", 
                    embedding_function=embeddings
                    )
#splitters
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

# splitter = CharacterTextSplitter(separator = " ", chunk_size=800, chunk_overlap=100)
# splitter = MarkdownTextSplitter(chunk_size=800, chunk_overlap=100)
# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

"""
here, this code for ingesting existing docs 
"""
def ingest_exi_doc():
    all_docs = []
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)
        if not os.path.isfile(file_path):
            continue
            #load the file based on file type
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(file_path)
            elif filename.lower().endswith(".docx"):
                loader = UnstructuredFileLoader(file_path)
            else:
                print(f"Skipping unsupported file: {filename}")
                continue
            #store the text to all_docs
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded {filename}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    if all_docs:
        chunks = splitter.split_documents(all_docs)
        vectorstore.add_documents(chunks)

ingest_exi_doc() 

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(docs_folder, file.filename)

    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_location)
        elif file.filename.lower().endswith(".txt"):
            loader = TextLoader(file_location)
        elif file.filename.lower().endswith(".docx"):
            loader = UnstructuredFileLoader(file_location)
        else:
            return JSONResponse({"message": "Unsupported file type."}, status_code=400)

        new_docs = loader.load()
        new_chunks = splitter.split_documents(new_docs)
        vectorstore.add_documents(new_chunks)

    except Exception as e:
        return JSONResponse({"message": f"Error ingesting file: {e}"}, status_code=500)

    return JSONResponse({"message": "File uploaded and ingested successfully!", "filename": file.filename})


@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    result = qa_chain.run(question)  
    return {"question": question, "answer": result}












# @app.post("/ask/")
# async def ask_question(question: str = Form(...)):
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#     llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)

#     prompt = ChatPromptTemplate.from_template("""
#     Answer the question using only the context below.
#     If the answer is not in context, say "I don’t know".

#     Context:
#     {context}
#     Question: {question}
#     Answer:
#     """)

#     rag_chain = (
#         {
#     "context": retriever,
#     "question": RunnablePassthrough()
#     }
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     answer = rag_chain.invoke(question)

#     return {"question": question, "answer": answer}









