import asyncio
import uuid

from src.agent.agent import build_the_batch_agent

async def main() -> None:
    agent = build_the_batch_agent()
    session_id = f"cli:{uuid.uuid4()}"

    print("Chat with agent (type 'exit' to quit).")

    while True:
        user_input = input("[USER] > ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        res = await agent.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print("[ASSISTANT] > ", res["output"])
        print()
        #-------
        print("[DEBUG]", res["intermediate_steps"])

if __name__ == "__main__":
    asyncio.run(main())
