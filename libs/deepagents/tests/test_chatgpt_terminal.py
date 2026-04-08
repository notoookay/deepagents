"""Terminal chat for testing ChatGPT Codex integration end-to-end.

Usage:
    uv run python tests/test_chatgpt_terminal.py
    uv run python tests/test_chatgpt_terminal.py --model gpt-5.1-codex
"""

import argparse
import asyncio

from deepagents import create_deep_agent


async def main(model: str) -> None:
    agent = create_deep_agent(model=f"chatgpt:{model}")
    config = {"configurable": {"thread_id": "test-terminal"}}

    print(f"ChatGPT Codex terminal test (model={model})")
    print("Type 'quit' to exit")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            print(f"\nAssistant: {result['messages'][-1].content}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.3-codex")
    args = parser.parse_args()
    asyncio.run(main(args.model))
