"""Entrypoint: `python -m rag`."""
import asyncio
import logging

from dotenv import load_dotenv

from .indexing import sync_documents
from .rag import Reply, respond

log = logging.getLogger(__name__)

SHOW_REASONING = True


def _print_reply(reply: Reply) -> None:
    if SHOW_REASONING and reply.reasoning:
        print("\n--- reasoning ---")
        print(reply.reasoning)
        print("--- /reasoning ---")
    print("\n=== Answer ===")
    print(reply.text)
    print()


async def _repl() -> None:
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            break

        try:
            kind, reply = await respond(query)
            log.info("Routed %r -> %s", query, kind)
            _print_reply(reply)
        except Exception:
            log.exception("Turn failed")
            print("(query failed — see log above)")


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    sync_documents()
    print("Type 'exit' (or Ctrl-D) to quit.")
    try:
        asyncio.run(_repl())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
