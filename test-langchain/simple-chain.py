from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

llm = OllamaLLM(model="qwen")
prompt = ChatPromptTemplate.from_messages([
    ("system", "现在你的名字叫小王."),
    ("user", "{input}")
])

chain = prompt | llm
response = chain.invoke("你是谁？")
print(response)