from __future__ import annotations
from typing import Optional, List, Union, Literal, Dict, Type

import os
import json
import logging
# from tqdm import tqdm
from pathlib import Path

from pydantic import BaseModel
from dataclasses import dataclass

from openai import AzureOpenAI, NOT_GIVEN

class StatusError(Exception):
    pass

@dataclass
class LLMResponse:
    text: str                # main output
    raw: dict                # full API response
    # optional fields
    logprobs: Optional[List] = None
    tokens: Optional[List[str]] = None

_client = None  # simple cache

# ---------------------
# JSON SCHEMA BUILDER
# ---------------------

def _enforce_no_additional_props(schema: dict) -> dict:
    """
    Recursively set additionalProperties=False for all objects.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)

        for key, value in schema.items():
            _enforce_no_additional_props(value)

    elif isinstance(schema, list):
        for item in schema:
            _enforce_no_additional_props(item)

    return schema


def pydantic_to_json_schema(
    model: Type[BaseModel],
    name: Optional[str] = None,
    strict: bool = True
) -> dict:
    
    """
    Convert a Pydantic model into an OpenAI-compatible JSON schema.

    Args:
        model (Type[BaseModel]):
            Pydantic model class.

        name (str):
            Name of the schema.

        strict (bool):
            Whether to enforce strict schema validation.

    Returns:
        dict:
            JSON schema formatted for OpenAI Responses API.
    """

    if name is None:
        name = model.__name__

    schema = model.model_json_schema()
    schema = _enforce_no_additional_props(schema)

    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": strict
            }
        }

# ----------------------------
# CLIENT AND AUTHENTIFICATION
# ----------------------------

def get_client(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
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
    return_metadata: bool = False,
    output_structure: Optional[Dict] = None
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

        output_structure (Dict | None):
            JSON schema defining a specific desired structured output. If None, output will default to free-form text.

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

    if output_structure is None:
        response = client.responses.create(
            model = deployment_model,
            input = prompt,
            temperature = temperature,
            max_output_tokens = max_tokens,
        )

    else:
        response = client.responses.create(
            model = deployment_model,
            input = prompt,
            temperature = temperature,
            max_output_tokens = max_tokens,
            text = output_structure #type: ignore
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
    batch_filename: Optional[Union[str, Path]] = None,
    id_key: str = "id",
    prompt_key: str = "prompt",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    output_structure: Optional[Dict] = None,
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
                "max_output_tokens": max_tokens,
                "text": output_structure
            }
        })

    if batch_filename:
        batch_filename = Path(batch_filename)
        batch_filename.parent.mkdir(parents = True, exist_ok = True)
        with open(batch_filename, "w", encoding = encoding) as f:
            for r in requests:
                f.write(json.dumps(r) + "\n")  # <-- each request is a line in JSONL
        return None

    return requests

def submit_batch(
    client: AzureOpenAI,
    batch_requests: Optional[List[dict]] = None,
    batch_input_path: Optional[Union[str, Path]] = None,
    encoding: str = "utf-8",
    verbosity: int = 1
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

    if batch_requests is not None:
        # Use temporary file
        batch_input_path = Path("temporary_batch_input.jsonl")
        batch_input_path.parent.mkdir(parents = True, exist_ok = True)

        # Write the batch requests to the file
        with open(batch_input_path, "w", encoding = encoding) as f:
            for r in batch_requests:
                f.write(json.dumps(r) + "\n")

    elif batch_input_path is not None:
        batch_input_path = Path(batch_input_path)
        if not batch_input_path.exists():
            raise ValueError(f"Batch input file not found: {batch_input_path}")

    else:
        raise ValueError("Must provide either batch_requests or batch_input_path.")

    # Upload batch input file
    with open(batch_input_path, "rb") as f:
        batch_file = client.files.create(file = f, purpose = "batch")

    # Create the batch job
    batch = client.batches.create(
        input_file_id = batch_file.id,
        endpoint = "/v1/responses",
        completion_window = "24h"
    )

    if verbosity > 0:
        print(f"Batch successfully submitted with id {batch.id}")

    return batch.id

def retrieve_batch_results(
    client: AzureOpenAI,
    batch_id: str,
    output_path: Optional[Union[str, Path]] = None,
    error_path: Optional[Union[str, Path]] = None,
    parse_json: bool = True,
    return_format: Literal["LLMResponse", "python"] = "LLMResponse",
    encoding: str = "utf-8",
    verbosity: int = 1
) -> dict:
    
    """
    Retrieve the results of a completed Azure OpenAI batch job.

    Downloads the batch output and error files (if any), saves them locally
    if paths are provided, and optionally parses the output into Python dictionaries
    or `LLMResponse` objects keyed by custom IDs.

    Args:
        client (AzureOpenAI):
            An initialized Azure OpenAI client.

        batch_id (str):
            The ID of the batch job to retrieve results from.

        output_path (str | Path | None):
            Optional path to save the batch output JSONL file.

        error_path (str | Path | None):
            Optional path to save the batch error JSONL file.

        parse_json (bool):
            If True, parse each line of the JSONL files into Python dictionaries.
            Defaults to True.

        return_format (Literal["LLMResponse", "python"]):
            Format of the returned results:
                - "python": returns raw parsed JSONL lines as lists.
                - "LLMResponse": returns dictionaries keyed by `custom_id`,
                  mapping to `LLMResponse` objects or error dictionaries.

        encoding (str):
            File encoding used when saving output/error files.

    Returns:
        dict:
            If return_format == "python":
                {
                    "outputs": list[dict],
                    "errors": list[dict]
                }

            If return_format == "LLMResponse":
                {
                    "outputs": dict[str, LLMResponse],
                    "errors": dict[str, dict]
                }

    Raises:
        ValueError:
            If the batch is not completed.

    Notes:
        - Requires that the batch status is "completed".
        - The `custom_id` field is used as the key for mapping outputs.
        - Assumes Responses API output structure.
        - Use `parse_json = False` to only save files without parsing.

    Example:
        >>> results = retrieve_batch_results(client, batch_id)
        >>> results["outputs"]["q1"].text
        'The propensity range is [0, 2]'
    """

    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise StatusError(f"Batch {batch_id} is not completed. Current status: {batch.status}")

    results = {"outputs": [], "errors": []}

    # ---- OUTPUT FILE ----
    if batch.output_file_id:
        output_content = client.files.content(batch.output_file_id).text

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents = True, exist_ok = True)
            with open(output_path, "w", encoding = encoding) as f:
                f.write(output_content)

        if parse_json:
            results["outputs"] = [
                json.loads(line) for line in output_content.strip().splitlines() if line
            ]

    # ---- ERROR FILE ----
    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id is not None:
        error_content = client.files.content(error_file_id).text

        if error_path:
            error_path = Path(error_path)
            error_path.parent.mkdir(parents = True, exist_ok = True)
            with open(error_path, "w", encoding = encoding) as f:
                f.write(error_content)

        if parse_json:
            results["errors"] = [
                json.loads(line) for line in error_content.strip().splitlines() if line
            ]

    total = len(results["outputs"]) + len(results["errors"])
    success = len(results['outputs'])
    errors = len(results['errors'])

    if verbosity > 0:
        print(
            f"Retrieved results for batch {batch_id}: "
            f"{success}/{total} ({round(success/total)}) successes, {errors}/{total} ({round(errors/total)}) errors."
        )

    # ---- RETURN FORMATS ----
    if return_format == "python":
        return results

    elif return_format == "LLMResponse":
        outputs_dict: Dict[str, LLMResponse] = {}
        errors_dict: Dict[str, dict] = {}

        # ---- Parse outputs ----
        for item in results["outputs"]:
            custom_id = item.get("custom_id")
            body = item.get("response", {}).get("body", {})

            texts = []
            logprobs_all = []

            for msg in body.get("output", []):
                for content in msg.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
                        if content.get("logprobs"):
                            logprobs_all.append(content.get("logprobs"))

            text = "\n".join(texts).strip()
            logprobs = logprobs_all if logprobs_all else None
            tokens = body.get("usage", None)

            outputs_dict[custom_id] = LLMResponse(
                text = text,
                raw = body,
                logprobs = logprobs,
                tokens = tokens
            )

        # ---- Parse errors ----
        for item in results["errors"]:
            custom_id = item.get("custom_id")
            errors_dict[custom_id] = item

        return {
            "outputs": outputs_dict,
            "errors": errors_dict
        }