# from langchain_community.document_loaders import UnstructuredHTMLLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# ## from langchain_community.embeddings import HuggingFaceEmbeddings   created an issue to update this import
# from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFacePipeline  # Updated import
# ##  from langchain_community.llms import HuggingFacePipeline
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from transformers import pipeline
# import os 

# ## here we write code to ensure the data directory exists
# if not os.path.exists("data"):
#     raise FileNotFoundError("Please create a 'data' folder and place 'mg-zs-warning-messages.html' in it")

# # load the HTML file 
# loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
# car_docs = loader.load()
# print(f"Loaded {len(car_docs)} documents from HTML file.")

# ## here what we do is split the documents into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50,length_function=len)
# chunks = text_splitter.split_documents(car_docs)
# print(f"Created {len(chunks)} chunks")

# ## Create embeddings  after perform the chunking 
# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
# print("Embedding model loaded. ")

# ## create a vector store using chroma 
# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     persist_directory="./chroma_db"  # Directory to persist the vector store persist means  here we save the vector store so that we can use it later without re-creating it
# )
# print("Vector store created and persisted.")

# ## here we make an model for generative model . 
# generator = pipeline(
#     "text-generation",
#     model="distilgpt2",
#     max_new_tokens=100,
#     truncation=True
# )
# llm =HuggingFacePipeline(pipeline=generator)
# print("Generative model loaded.")

# ## here we create a prompt template for the chatbot 
# prompt_template = """Use the following pieces of context to answere the question about the car warning messages. provide a clear and concise response based  only on the context . If you dont know the answere,say so.
# Context: {context}
# Question: {question}
# Answer:   """
# prompt = ChatPromptTemplate.from_template(prompt_template)

# ## create RAG chain for the chatbot 
# retriever = vector_store.as_retriever(search_kwargs={"k":3}) ## k is the number of documents to retrieve and use for the answer
# rag_chain = (
#     {"context":retriever, "question":RunnablePassthrough()}
#     | prompt
#     | llm
# )
# print("RAG chain created.")

# # Test query 
# query = "What should I do if I see 'Engine coolant temperature high'?"
# response = rag_chain.invoke(query)
# print("Query:",query)
# print("Response:",response.strip())   ## use strip to remove any leading or trailing whitespace


from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
import os

# Ensure the data directory exists
if not os.path.exists("data"):
    raise FileNotFoundError("Please create a 'data' folder and place 'mg-zs-warning-messages.html' in it")

# Load the HTML file
loader = UnstructuredHTMLLoader(file_path="data/mg-zs-warning-messages.html")
car_docs = loader.load()
print(f"Loaded {len(car_docs)} document(s)")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len,
)
chunks = text_splitter.split_documents(car_docs)
print(f"Created {len(chunks)} chunks")

# Create embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
print("Embedding model loaded")

# Create vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector store created and saved")

# Create generative model
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=50,  ## reduced for conciseness means shorter responses
    truncation=True,
    pad_token_id=50256,   # set pad_token_id explictly for gpt2
    temperature=0.7,  # set temperature for more controlled responses
    top_p=0.9,  # set top_p for nucleus sampling means more diverse responses 

)
llm = HuggingFacePipeline(pipeline=generator)
print("Generative model loaded")

# Define prompt template
prompt_template = """Use the following context to answer the question about car warning messages. Provide a clear and concise response based only on the context. If you don't know the answer, say so.
Context: {context}
Question: {question}
Answer: """
prompt = ChatPromptTemplate.from_template(prompt_template)

# Create RAG chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})    ## increased k from 3 to 5 for better coverage 
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | (lambda x: x.strip().split("Answer:")[-1].strip())   ## strip and split to get the answer part only
)

# Test query
query = "What should I do if I see 'Engine Coolant Temperature High'?"
response = rag_chain.invoke(query)
print("Query:", query)
print("Response:", response)