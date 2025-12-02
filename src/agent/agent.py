from __future__ import annotations

from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor


from src.config import config
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.vector_db.chroma_store import TheBatchChromaIndexer

from src.agent.prompt import SYSTEM_PROMPT
from src.agent.tools import make_multimodal_search_tool

HISTORY_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in HISTORY_STORE:
        HISTORY_STORE[session_id] = InMemoryChatMessageHistory()
    return HISTORY_STORE[session_id]

def build_the_batch_agent() -> RunnableWithMessageHistory:
    """
    Build a LangChain agent with:
      - Gemini LLM
      - TheBatchChromaIndexer-backed search tool
      - Chat history via RunnableWithMessageHistory
    """
    text_embedder = TextEmbedder(
        model_name=config.text_embed_model_name,
        device=config.device,
    )

    clip_embedder = ImageEmbedder(
        model_name=config.clip_model_name,
        device=config.device,
    )

    indexer = TheBatchChromaIndexer(
        text_embedder=text_embedder,
        clip_embedder=clip_embedder,
    )

    tools = [make_multimodal_search_tool(indexer)]

    llm = ChatGoogleGenerativeAI(
        model=config.gemini_model_name,
        temperature=0.4,
        google_api_key=config.google_ai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )

    return agent_with_history
