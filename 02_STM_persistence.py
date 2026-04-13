from langgraph.graph import StateGraph, START, END, MessagesState
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langgraph.graph.message import add_messages
import os
load_dotenv()
DB_URI = os.environ["DB_URI"]

llm = HuggingFaceEndpoint(
  repo_id="Qwen/Qwen3-Coder-Next",
  task='text-generation'
)

model = ChatHuggingFace(llm=llm)

def call_model(state:MessagesState):
  response = model.invoke(state['messages'])
  return {"messages":[response]}

builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Run ONCE (creates tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # Thread 1 (remembers)
    t1 = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Nitish"}]}, t1)
    out1 = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, t1)
    print("Thread-1:", out1["messages"][-1].content)
    
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Run ONCE (creates tables)
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    # Thread 2 (fresh)
    t2 = {"configurable": {"thread_id": "thread-2"}}
    out2 = graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, t2)
    print("Thread-2:", out2["messages"][-1].content)
    
    
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres"
t1 = {"configurable": {"thread_id": "thread-1"}}

with PostgresSaver.from_conn_string(DB_URI) as cp:
    g = builder.compile(checkpointer=cp)

    snap = g.get_state(t1)  # <-- pulls from Postgres
    msgs = snap.values.get("messages", [])
    print("Last message:", msgs[-1].content if msgs else None)