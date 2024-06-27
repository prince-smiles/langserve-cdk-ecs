from fastapi import FastAPI
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .agents.website_bot import validate_and_ask_next
from typing import Any
from .agents.website_bot import website_chat_agent


app = FastAPI()

# Define input and output models for FastAPI
class ChatInput(BaseModel):
    session_id: str
    user_response: str = None

class ChatOutput(BaseModel):
    chatbot_response: str

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

add_routes(
    app,
    website_chat_agent.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
    path="/website-bot",
)

# For health check, otherwise this will return 404
@app.get("/")
def get_root():
    return {"message": "FastAPI running in a Docker container"}

@app.post("/chat", response_model=ChatOutput)
def chat_endpoint(input: ChatInput):
    session_id = input.session_id
    user_response = input.user_response or ""
    
    chatbot_response = validate_and_ask_next(session_id, user_response)
    print("Res:", chatbot_response)
    return ChatOutput(chatbot_response=chatbot_response)
    # return chatbot_response

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
