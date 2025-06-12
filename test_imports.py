# # # try:
# # #     from langchain_community.document_loaders import UnstructuredHTMLLoader
# # #     print("UnstructuredHTMLLoader imported")
# # #     from langchain_text_splitters import RecursiveCharacterTextSplitter
# # #     print("RecursiveCharacterTextSplitter imported")
# # #     from langchain_community.embeddings import HuggingFaceEmbeddings
# # #     print("HuggingFaceEmbeddings imported")
# # #     from langchain_chroma import Chroma
# # #     print("Chroma imported")
# # #     from langchain_community.llms import HuggingFacePipeline
# # #     print("HuggingFacePipeline imported")
# # #     from langchain_core.prompts import ChatPromptTemplate
# # #     print("ChatPromptTemplate imported")
# # #     from langchain_core.runnables import RunnablePassthrough
# # #     print("RunnablePassthrough imported")
# # #     from transformers import pipeline
# # #     print("Transformers pipeline imported")
# # # except ImportError as e:
# # #     print(f"Import error: {e}")

# # try:
# #     from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# #     print("langchain_huggingface imports OK")
# #     from langchain_community.document_loaders import UnstructuredHTMLLoader
# #     print("UnstructuredHTMLLoader OK")
# #     from langchain import __version__
# #     print(f"langchain version: {__version__}")
# #     from langchain_core import __version__ as core_version
# #     print(f"langchain-core version: {core_version}")
# # except ImportError as e:
# #     print(f"Import error: {e}")

# try:
#     from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
#     print("langchain_huggingface imports OK")
#     from langchain_community.document_loaders import UnstructuredHTMLLoader
#     print("UnstructuredHTMLLoader OK")
#     from langchain import __version__
#     print(f"langchain version: {__version__}")
#     from langchain_core import __version__ as core_version
#     print(f"langchain-core version: {core_version}")
# except ImportError as e:
#     print(f"Import error: {e}")


try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    print("langchain_huggingface imports OK")
    from langchain_community.document_loaders import UnstructuredHTMLLoader
    print("UnstructuredHTMLLoader OK")
    from langchain import __version__
    print(f"langchain version: {__version__}")
    from langchain_core import __version__ as core_version
    print(f"langchain-core version: {core_version}")
    from langchain_core.caches import BaseCache
    print("BaseCache OK")
except ImportError as e:
    print(f"Import error: {e}")