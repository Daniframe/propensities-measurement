from __future__ import annotations
from typing import Optional, List, Union

import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

from openai import AzureOpenAI

@dataclass
class LLMResponse:
    text: str                # main output
    raw: dict                # full API response
    # optional fields
    logprobs: dict | None = None
    tokens: List[str] | None = None

_client = None  # simple cache

# ----------------------------
# CLIENT AND AUTHENTIFICATION
# ----------------------------

def get_client(
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str = "2025-03-01-preview", # First version to have responses API
) -> AzureOpenAI:
    
    """
    Create and return a cached Azure OpenAI client.

    This function initializes a singleton Azure OpenAI client using either
    explicitly provided credentials or environment variables. Subsequent
    calls return the same client instance to avoid unnecessary re-creation.

    Configuration priority:
        1. Function arguments (`api_key`, `endpoint`, `api_version`)
        2. Environment variables (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`)
        3. Default API version ("2024-12-01-preview")

    Args:
        api_key (str | None):
            Azure OpenAI API key. If not provided, the value is read from
            the `AZURE_OPENAI_API_KEY` environment variable.

        endpoint (str | None):
            Azure OpenAI endpoint URL. If not provided, the value is read
            from the `AZURE_OPENAI_ENDPOINT` environment variable.

        api_version (str):
            Azure OpenAI API version to use. Defaults to
            "2024-12-01-preview".

    Returns:
        AzureOpenAI:
            An initialized and cached Azure OpenAI client.

    Raises:
        ValueError:
            If no API key or endpoint is provided via arguments or
            environment variables.

    Notes:
        - The client is cached after the first initialization. If different
          credentials are passed in subsequent calls, they will be ignored.
        - To use multiple clients with different configurations, avoid
          caching or modify this function accordingly.
        - Designed for script and pipeline usage where a single shared
          client is sufficient.

    Example:
        >>> client = get_client()
        >>> response = client.chat.completions.create(...)

        >>> # Override environment variables
        >>> client = get_client(api_key="...", endpoint="...")
    """

    global _client

    if _client is not None:
        return _client

    if api_key is None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if endpoint is None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY environment variable")

    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT environment variable")

    _client = AzureOpenAI(
        azure_endpoint = endpoint,
        api_key = api_key,
        api_version = api_version,
    )

    return _client


# -------------------------
# AZURE INFERENCE (SINGLE)
# -------------------------

def llm_single_response(
    client: AzureOpenAI,
    deployment_model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    return_metadata: bool = False
) -> str | LLMResponse:
    """
    Generate text using Azure OpenAI Responses API.

    This function sends a single-shot prompt to a given Azure OpenAI client
    and returns either a simple string output or a dataclass containing
    full metadata.

    Args:
        client (AzureOpenAI):
            An initialized Azure OpenAI client. This allows explicit control
            over API keys, endpoint, and version outside the function.

        deployment_model (str):
            Model name to use for generation.

        prompt (str):
            The prompt to send to the LLM.

        temperature (float):
            Sampling temperature. Defaults to 0.0 for deterministic output.

        max_tokens (int | None):
            Maximum number of tokens to generate. If None, defaults to model's limit.

        return_metadata (bool):
            If True, returns an `LLMResponse` dataclass with text, raw response,
            logprobs, and tokens. If False (default), returns only the generated text.

    Returns:
        str | LLMResponse:
            Generated text as a string if `return_metadata=False`.
            Otherwise, an `LLMResponse` dataclass with full metadata.

    Notes:
        - Does not instantiate a client internally; the client must be passed.
        - Suitable for both sequential execution and integration with batch pipelines.
        - Ensures deterministic behavior when `temperature = 0.0`.

    Example:
        >>> client = get_client()
        >>> text = generate_text(client, prompt = "Hello, world!")

        >>> # Get full metadata
        >>> resp = generate_text(client, prompt = "Hello, world!", return_metadata = True)
        >>> print(resp.text)
        >>> print(resp.raw)
    """
    response = client.responses.create(
        model = deployment_model,
        input = prompt,
        temperature = temperature,
        max_output_tokens = max_tokens,
    )

    output_text = response.output_text.strip() if hasattr(response, "output_text") else ""

    result = LLMResponse(
        text = output_text,
        raw = response.model_dump(),
        logprobs = getattr(response, "logprobs", None),
        tokens = getattr(response, "tokens", None)
    )

    return result if return_metadata else result.text


# ------------------------
# AZURE INFERENCE (BATCH)
# ------------------------
def create_batch_requests(
    prompts: List[dict],
    deployment_model: str,
    batch_filename: Optional[Path] = None,
    id_key: str = "id",
    prompt_key: str = "prompt",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    encoding: str = "utf-8"
) -> List[dict] | None:
    
    """
    Build batch request payloads for Azure OpenAI Responses API.

    This function prepares a list of JSON-compatible request dictionaries,
    ready to be written to a JSONL file and submitted as a batch to Azure OpenAI.
    Each question must already have its prompt constructed (e.g., with system
    instructions and rubric).

    Args:
        prompts (list[dict]):
            A list of prompts, each represented as a dictionary with:
                - `id_key` (str): unique identifier for the question
                - `prompt_key` (str): fully formatted prompt text

        deployment_model (str):
            Name of the LLM model to use for all requests.

        batch_filename (Path):
            Optional file path to save the batch as a .jsonl file.

        id_key (str):
            Field name in `prompts` dict as the unique identifier for each prompt

        prompt_key (str):
            Field name in `prompts` dict as the prompt text

        temperature (float):
            Sampling temperature for generation.

        max_tokens (int | None):
            Optional maximum number of tokens to generate for each request.

    Returns:
        list[dict] | None:
            If `batch_filename` is not provided, returns a list of dictionaries representing individual batch requests, each including 'custom_id', 'method', 'url', and 'body'.
            If `batch_filename` is provided, the requests are written into a file and this function returns nothing

    Notes:
        - The 'custom_id' of each request is set to the question's 'id' field.
        - Prompts should already include any "system instructions" or formatting.
        - The returned list can be written directly to a JSONL file for batch submission.

    Example:
        >>> questions = [{"id": "q1", "prompt": "Your prompt here"}]
        >>> requests = create_batch_requests(questions)
        >>> len(requests)
        1
        >>> requests[0]["custom_id"]
        'q1'
    """

    requests = []
    for q in prompts:
        requests.append({
            "custom_id": q[id_key],
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": deployment_model,
                "input": q[prompt_key],
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        })

    if batch_filename:
        batch_filename.parent.mkdir(parents = True, exist_ok = True)
        with open(batch_filename, "w", encoding = encoding) as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")  # <-- each request is a line in JSONL
        return None

    return requests

def submit_batch(
    client: AzureOpenAI,
    batch_requests: Optional[List[dict]] = None,
    batch_input_path: Optional[Path] = None,
    encoding: str = "utf-8"
) -> str:
    
    """
    Submit a batch job to Azure OpenAI Responses API.

    This function either uploads a pre-built batch input file (JSONL) or a list
    of batch requests, and creates an asynchronous batch job using Azure OpenAI.
    The function returns the batch ID for tracking and later retrieval.

    Args:
        client (AzureOpenAI):
            An initialized Azure OpenAI client.

        batch_requests (list[dict] | None):
            Optional. A list of batch request dictionaries, as returned by
            `create_batch_requests()`. Each dictionary must include:
                - 'custom_id': unique identifier
                - 'method': "POST"
                - 'url': "/v1/responses"
                - 'body': dictionary with model, prompt, temperature, etc.

        batch_input_path (Path | None):
            Optional. Path to a pre-existing JSONL batch input file. If provided,
            `batch_requests` will be ignored.

    Returns:
        str:
            The Azure batch ID of the submitted job.

    Raises:
        ValueError:
            If neither `batch_requests` nor `batch_input_path` is provided,
            or if `batch_input_path` does not exist.

    Notes:
        - If `batch_requests` is provided, the function writes it to
          `batch_input_path` (or "temporary_batch_input.jsonl" by default) for upload.
        - Designed to support large datasets and asynchronous batch execution.

    Example:
        >>> client = get_client()
        >>> batch_requests = create_batch_requests(questions)
        >>> batch_id = submit_batch(client, batch_requests=batch_requests)
        >>> print(batch_id)
    """

    if batch_input_path is None and batch_requests is None:
        raise ValueError("Must provide either batch_requests or batch_input_path.")

    # If batch_requests is provided, write to file
    if batch_requests is not None:
        if batch_input_path is None:
            batch_input_path = Path("temporary_batch_input.jsonl")
        batch_input_path.parent.mkdir(parents = True, exist_ok = True)
        with open(batch_input_path, "w", encoding = encoding) as f:
            for r in batch_requests:
                f.write(json.dumps(r) + "\n")

    # If batch_input_path is provided, ensure it exists
    if batch_input_path:
        if not batch_input_path.exists():
            raise ValueError(f"Batch input file not found: {batch_input_path}")

        # Upload batch input file
        with open(batch_input_path, "rb") as f:
            batch_file = client.files.create(file = f, purpose = "batch")

    # Create the batch job
    batch = client.batches.create(
        input_file_id = batch_file.id,
        endpoint = "/v1/responses",
        completion_window = "24h"
    )

    return batch.id