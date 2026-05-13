from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama2")

llm.invoke("how can langsmith help with testing?")