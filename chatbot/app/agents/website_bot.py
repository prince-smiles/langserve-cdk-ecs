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
from collections import defaultdict


questions = [
    {"question": "Hello! What is your name?", "validation_tool": "validate_name"},
    {"question": "How can I help you today?", "validation_tool": "validate_issue"},
    {"question": "Can you provide more details about your issue?", "validation_tool": "validate_details"},
    {"question": "Is there anything else you would like to add?", "validation_tool": "validate_additional_info"}
]

uri = os.environ['MONGO_CONNECTION_STRING']
client = MongoClient(uri, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client[os.environ['MONGO_DATABASE']]
embeddings_collection = db['scrapped_data']

def vector_search(query, top_k=5):
    embedding = OpenAIEmbeddings().embed_documents([query])[0]
    search_result = embeddings_collection.aggregate([
        {
            "$search": {
                "knnBeta": {
                    "vector": embedding,
                    "path": "embedding",
                    "k": top_k
                }
            }
        }
    ])
    documents = []
    for result in search_result:
        doc = Document(page_content=result["page_content"], metadata={"url": result["url"]})
        documents.append(doc)
    return documents

def validate_name(input_text: str) -> bool:
    return len(input_text.strip()) > 0

def validate_issue(input_text: str) -> bool:
    return len(input_text.strip()) > 0

def validate_details(input_text: str) -> bool:
    return len(input_text.strip()) > 0

def validate_additional_info(input_text: str) -> bool:
    return True  # Accept any additional info

# Create tool instances
validate_name_tool = Tool(
    name="validate_name",
    func=validate_name,
    description="Validates if the provided name is valid."
)

validate_issue_tool = Tool(
    name="validate_issue",
    func=validate_issue,
    description="Validates if the provided issue description is valid."
)

validate_details_tool = Tool(
    name="validate_details",
    func=validate_details,
    description="Validates if the provided details are valid."
)

validate_additional_info_tool = Tool(
    name="validate_additional_info",
    func=validate_additional_info,
    description="Validates if the additional info is valid."
)

# tools = [validate_name_tool, validate_issue_tool, validate_details_tool, validate_additional_info_tool]
session_states = defaultdict(lambda: {"current_question": 0, "answers": []})

def qa_chain_tool(query: str) -> str:
    results = vector_search(query)
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

tools = [ qa_chain_tool, validate_name_tool, validate_issue_tool, validate_details_tool, validate_additional_info_tool ]

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

def validate_and_ask_next(session_id: str, user_response: str) -> str:
    state = session_states[session_id]
    print("Inside Validate:", state)

    current_question_index = state["current_question"]
    
    if current_question_index > 0:
        # Validate the previous answer
        tool_name = questions[current_question_index - 1]["validation_tool"]
        validation_tool = next(tool for tool in tools if tool.name == tool_name)
        is_valid = validation_tool.func(user_response)
        if not is_valid:
            return f"Invalid response. {questions[current_question_index - 1]['question']}"
        
        state["answers"].append(user_response)
    
    # Move to the next question
    if current_question_index < len(questions):
        next_question = questions[current_question_index]["question"]
        state["current_question"] += 1
        return next_question
    else:
        return "Thank you for the information! We will get back to you shortly."

