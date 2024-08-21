from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from dotenv import dotenv_values
config = dotenv_values(".env")

os.environ["OPENAI_API_KEY"] = config["openai_api"]
os.environ["ANTHROPIC_API_KEY"] = config["ANTHROPIC_API_KEY"]

def model():
    
    dataFamily = f"./Data/"
    vectordb = Chroma(persist_directory=dataFamily, embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()

    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",
                        temperature=0.9,
                        max_tokens_to_sample= 1500
                        )
    
    system_template = """
    system: คุณคือผู้ช่วยที่ตอบคำถามภายในสถาบันเทคโนโลยีจิตรลดาของเรา คุณจะตอบคถามที่มีประโยชน์ต่อนักศึกษามากที่สุด และ เมื่อสิ้นสุดประโยคคุณต้องลงท้ายด้วย ``ค่ะ``\n
    """

    Template = """
    กรุณาตอบตาม Context นี้เพื่อสร้างคำตอบที่มีประโยชน์สูงสุดให้แก่ นักศึกษาของเรา: {context}

    Question: {question}
    """

    template = system_template + Template

    prompt = ChatPromptTemplate.from_template(template)

    setup_and_retrieval = RunnableParallel(
        {"context": retriever,
        "question": RunnablePassthrough()}
        )
    output_parser = StrOutputParser()
    planer_chain = setup_and_retrieval | prompt | llm | output_parser
    return planer_chain



if __name__ == "__main__":
    # user_input = str(input("กรุณาถามคำถาม: "))
    res = model()
    while True:
        user_input = str(input("\nกรุณาถามคำถาม: "))
        for chunks in res.stream(user_input):
            print(chunks, end="", flush=True)