from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor, Tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from ..tools.rag import get_treatment_price
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import numpy as np
import faiss

uri = os.environ['MONGO_CONNECTION_STRING']
client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client[os.environ['MONGO_DATABASE']]
embeddings_collection = db['scrapped_data']
index = faiss.IndexFlatL2(d)

def vector_search_faiss(query, top_k=5):
    embedding = OpenAIEmbeddings().embed_documents([query])[0]
    embedding = np.array([embedding])  # Convert to numpy array
    distances, indices = index.search(embedding, top_k)
    
    results = []
    for idx in indices[0]:
        result = embeddings_collection.find_one({"embedding": document_embeddings[idx].tolist()})
        results.append(Document(page_content=result["page_content"], metadata={"url": result["url"]}))
    return results


def qa_chain_tool(query: str) -> str:
    results = vector_search_faiss(query)
    if results:
        source_doc = results[0]
        text = source_doc.page_content
        url = source_doc.metadata['url']
        return f"Text: {text}\nURL: {url}"
    return "No relevant information found"

qa_chain_tool = Tool(
    name="qa_chain_tool",
    func=qa_chain_tool,
    description="Answers questions based on the content of the website"
)

system_prompt = """
You are a Dental Assistant named Rahul, People ask you about price of dental treatments and you reply
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

tools = [ qa_chain_tool ]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# print(os.environ["MONGO_DATABASE"])
# print(os.environ["MONGO_CONNECTION_STRING"])

website_chat_agent = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=os.environ["MONGO_CONNECTION_STRING"],
        database_name=os.environ["MONGO_DATABASE"],
        collection_name=os.environ["MONGO_COLLECTION"],
    ),
    input_messages_key="input",
    history_messages_key="chat_history",
)
