#!/usr/bin/env python3
"""Query apple/LensVLM-9B through its hosted Hugging Face demo Space.

LensVLM renders a long text document into compressed page images, scans them,
and expands only the pages it needs to answer a question. The 9B model weights
are ~19 GB and need a GPU, so this client calls the hosted Space instead of
loading the model locally.

Usage:
    pip install -r requirements.txt
    export HF_TOKEN=hf_...          # optional; avoids anonymous rate limits
    python lensvlm_client.py --file contract.txt --question "What is the retention cap?"
"""
import argparse
import os
import sys

from gradio_client import Client

DEFAULT_SPACE = "hugging-apps/lensvlm-9b-demo"
DEFAULT_API = "/run_lensvlm"


def ask(document, question, compression="10x", max_turns=4, max_new_tokens=512,
        space=DEFAULT_SPACE, api_name=DEFAULT_API):
    client = Client(space, hf_token=os.environ.get("HF_TOKEN"))
    return client.predict(
        document=document,
        question=question,
        compression=compression,
        max_turns=max_turns,
        max_new_tokens=max_new_tokens,
        api_name=api_name,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="Plain-text document (use '-' for stdin)")
    parser.add_argument("--question")
    parser.add_argument("--compression", choices=["5x", "10x", "15x"], default="10x")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--api-name", default=DEFAULT_API)
    parser.add_argument("--list-api", action="store_true", help="Print the Space's endpoints and exit")
    args = parser.parse_args()

    if args.list_api:
        Client(args.space, hf_token=os.environ.get("HF_TOKEN")).view_api()
        return
    if not args.file or not args.question:
        parser.error("--file and --question are required")

    document = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    result = ask(document, args.question, args.compression, args.max_turns,
                 args.max_new_tokens, args.space, args.api_name)
    print(result)


if __name__ == "__main__":
    main()
