from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,GenerationConfig
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
import torch

# 定义文件所在的路径
DOC_PATH = "./docs"

# 使用 DirectoryLoader 从指定路径加载文件。"*.md" 表示加载所有 .md 格式的文件，这里仅导入文章 10（避免文章 20 的演示内容对结果的影响）
loader = DirectoryLoader(DOC_PATH, glob="test.md")

# 加载目录中的指定的 .md 文件并将其转换为文档对象列表
documents = loader.load()

# 文本处理
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 尝试调整它
    chunk_overlap=100,  # 尝试调整它
    #length_function=len,  # 可以省略
    #separators=["\n\n", "\n", " ", "。", ""]  # 可以省略
)
docs = text_splitter.split_documents(documents)

# 生成嵌入（使用 Hugging Face 模型）
# 指定要加载的预训练模型的名称，参考排行榜：https://huggingface.co/spaces/mteb/leaderboard
model_name = "./Qwen3-Embedding-0.6B"

# 创建 Hugging Face 的嵌入模型实例，这个模型将用于将文本转换为向量表示（embedding）
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

# 建立向量数据库
vectorstore = FAISS.from_documents(docs, embedding_model)

# 保存向量数据库（可选）
#vectorstore.save_local("faiss_index")

# 加载向量数据库（可选）
# 注意参数 allow_dangerous_deserialization，确保你完全信任需要加载的数据库（当然，自己生成的不需要考虑这一点）
#vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 加载文本生成模型
# 本地
model_path = './Qwen3.5-0.8B'
# 远程
#model_path = 'neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit'

# 加载
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

gen_config = GenerationConfig(
    max_length=4096,    # 指定生成文本的最大长度
    pad_token_id=tokenizer.eos_token_id
)

# 创建文本生成管道
generator = pipeline(
    "text-generation",  # 指定任务类型为文本生成
    model=model,
    tokenizer=tokenizer,
    # max_length=4096,    # 指定生成文本的最大长度
    # pad_token_id=tokenizer.eos_token_id
    # generation_config=gen_config
)

# 包装为 LangChain 的 LLM 接口
llm = HuggingFacePipeline(pipeline=generator)

custom_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

# 构建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",   # 直接堆叠所有检索到的文档
    retriever=retriever,  # 使用先前定义的检索器来获取相关文档
    # chain_type_kwargs={"prompt": custom_prompt}  # 可以选择传入自定义提示模板（传入的话记得取消注释），如果不需要可以删除这个参数
)

# 提出问题
query = "西游记是什么？"

# 获取答案
answer = qa_chain.invoke(query)
print(answer['result'])