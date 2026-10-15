from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

SYSTEM_PROMPT = """You are a Fang Yuan from novel Reverend Insanity.
You're helpful and all other traits of Fang Yuan's personality.
"""

model = init_chat_model(
    "gemini-3.5-flash-lite",
    model_provider="google-genai",
    temperature=0.9,
    timeout=600,
    max_tokens=3000,
    streaming=True,
)

checkpointer = InMemorySaver()
config={"configurable": {"thread_id": "FangYuanChat"}}

agent = create_agent(
    model=model,
    tools=[Add, Sub, Div],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)

content = "what are you doing here?"
result = agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config=config,
)

print(result["messages"][-1].content_blocks)