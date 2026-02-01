from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    "给我一句关于 {topic} 的科普说明"
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "黑洞"})
print(result)