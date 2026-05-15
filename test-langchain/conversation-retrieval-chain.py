
import os
os.environ["USER_AGENT"] = "MyCoolApp/1.0"

from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 第一步： 准备LLM，Embedding模型，读取数据，数据处理
llm = OllamaLLM(model="qwen:1.8b")
llm.invoke("langsmith是做什么的？")
loader = WebBaseLoader("https://docs.langchain.com/langsmith/home")
docs = loader.load()

embeddings = OllamaEmbeddings(model="qwen:1.8b") #模型默认为llama2
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

# 第二步：创建检索Chain，通过LLM根据历史对话记录检索相关内容
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "根据上述对话，生成一个搜索查询以查找与对话相关的信息")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# 动作确认：用来确认retriever_chain的执行情况，执行检索，不会真的回答问题"告诉我怎么做"
# chat_history = [HumanMessage(content="LangSmith的网址？"), AIMessage(content="可以！")]
# response1 = retriever_chain.invoke({
#    "chat_history": chat_history,
#    "input": "告诉我怎么做"
# })
# print(response1)

# 第三步：创建检索chain，根据检索出的内容，历史对话，与用户的问题产生回答
prompt = ChatPromptTemplate.from_messages([
    ("system", "根据下面的上下文回答用户的问题：\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# 第四步：代码执行与测试
chat_history = [HumanMessage(content="LangSmith能帮助测试我的LLM应用吗？"), AIMessage(content="可以！")]
response = retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "告诉我怎么做"
})
print(response["answer"])