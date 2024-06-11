from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from ..tools.rag import get_treatment_price
import os


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

tools = [get_treatment_price]

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
