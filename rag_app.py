import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



#LOADING THE ENVIRONMENT VARIABLES (API KEY)
load_dotenv()


#CREATING EMBEDDINGS AND INITIALIZING THE VECTOR DB
#we convert the chunks into vector coordinates and save them locally
#the VectorDB will then be used later on for fast lookups
print("Creating vector database...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


#check if we have already built a Vector DB -> if yes, we don't need to rebuilt it as it uses a lot of resources
if(os.path.exists("./chroma_db")):
    print("Loading existing vector database ...")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
    #the Vector DB doesn't exist yet -> built it

    #LOADING THE PDF
    #PyPDFLoader removes all lay-out formatting and extracts the raw text into a format LangChain can work with
    print("Loading PDF document...")
    loader = PyPDFLoader("./data/Vives_Examenreglement.pdf")
    document = loader.load()
    #print(document[0].page_content)


    #CHUNKING THE TEXT
    #this splits the text into 'chunks' so we don't always send the entire PDF to the LLM later on
    #chunk_size = 500 -> chunks with a size of 500 characters
    #chunk_overlap = 50 -> new chunk repeats the last 50 characters of the previous chunk to preserve context between them
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(document)
    #print(chunks[4].page_content)


    #EMBEDDING THE CHUNKS
    print("Embedding the chunks (might take a while, free tier limit) ...")
    #the code below is correct, but causes the 429 ERROR (RESOURCE_EXHAUSTED) because of the free tier we're using on Gemini
    # vector_db = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     persist_directory="./chroma_db"
    # )

    #so instead we use the following code, which embeds the chunks in batches and waits a time in between
    #this makes sure we can keep working with the free tier
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vector_db.add_documents(chunks[i:i+batch_size])
        print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")
        if(i + batch_size < len(chunks)):
            time.sleep(35)



#SETTING UP RETRIEVER AND PROMPT TEMPLATE
#convert the DB into a retriever object that can search through the stored document embeddings and return the most relevant chunks
#k=3 -> return only the 3 most relevant chunks for any given question (keeps things clean and efficient)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

#prompt template are hidden instructions for the model
#LangChain automatically replaces context with the retrieved relevant chunks
#LangChain automatically replaces the question with the user's actual query
#it's important to engineer a good prompt, so the answer stays correct, concise, polite and no hallucinations occur
template = """
You are a helpful assistant that answers questions about the VIVES exam regulations.
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer in the language of the question.
Use three sentences maximum and keep the answer concise and polite.

Context: {context}

Question: {question}

Answer:
"""

prompt = PromptTemplate.from_template(template)


#INITIALIZING THE LLM & CONSTRUCTING THE RAG CHAIN
#we choose a model from Google Gemini AI
#temperate=0 -> extra setting which forces the model to be completely factual/analytical rather than creative
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

#join the content of each chosen chunk together into 1 text
#this is needed to pass it to the LLM
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

#connect everything together into a RAG chain pipeline
#when rag_chain is invoked with a question, it gets embedded and the retriever fetches the most similar chunks
#format_docs joins the chunks into a single string
#the context (chunks) and question are inserted into the prompt's {context} and {question}
#the filled in prompt is sent to the LLM which generates an answer
#the answer is parsed into a message object (human-readable) 
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
     | prompt
     | llm
     | StrOutputParser()
)


#INVOKING THE RAG CHAIN WITH A QUESTION
while True:
    q = input("Ask a question (or 'exit'): ")
    if(q.lower() == "exit"):
        break

    print(f"Answer: {rag_chain.invoke(q)}")


