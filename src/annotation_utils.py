from __future__ import annotations
from typing import Optional, List, Dict, Union, Type, Literal, Tuple

import os
import re
import time
import json
import logging
from tqdm import tqdm

from pathlib import Path
from pydantic import BaseModel
from dataclasses import dataclass

import pandas as pd

from openai import AzureOpenAI
from . import azure_utils as azutils
# import azure_utils as azutils
LLMResponse = azutils.LLMResponse

# Pattern emitted by the presentation prompt in `rubrics/presentation.txt`
FINAL_RANGE_PATTERN = r"<FINAL_RANGE>\s*\[\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\]\s*</FINAL_RANGE>"

class PropAnnotationSchema(BaseModel):
    lower_bound: float
    upper_bound: float
    explanation: str

@dataclass(slots = True)
class PropensityAnnotation:
    propensity: str
    system_prompt: str
    rubric: str
    task_prompt: str
    source: str
    presentation_prompt: str = "Now annotate the following instance:\n"
    llm_response: Optional[Union[Dict, LLMResponse]] = None
    lower_bound: Optional[int | float] = None
    upper_bound: Optional[int | float] = None
    metadata: Optional[Dict] = None

    def __init__(
        self,
        propensity: str,
        system_prompt: str,
        rubric: str,
        task_prompt: str,
        source: str,
        presentation_prompt: str = "Now annotate the following instance:\n",
        llm_response: Optional[Union[Dict, LLMResponse]] = None,
        lower_bound: Optional[int | float] = None,
        upper_bound: Optional[int | float] = None,
        metadata: Optional[Dict] = None
    ):
        
        self.propensity = propensity
        self.system_prompt = system_prompt
        self.rubric = rubric
        self.task_prompt = task_prompt
        self.source = source
        self.llm_response = llm_response
        self.presentation_prompt = presentation_prompt
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.metadata = metadata

        if isinstance(llm_response, LLMResponse):
            self.llm_response = llm_response
        elif isinstance(llm_response, dict):
            texts = []
            logprobs_all = []

            for msg in llm_response.get("output", []):
                for content in msg.get("content", []):
                    if content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
                        if content.get("logprobs"):
                            logprobs_all.append(content.get("logprobs"))

            text = "\n".join(texts).strip()
            logprobs = logprobs_all if logprobs_all else None
            tokens = llm_response.get("usage", None)

            self.llm_response = LLMResponse(
                text = text,
                raw = llm_response,
                logprobs = logprobs,
                tokens = tokens
            )

    def get_full_prompt(self) -> str:
        return self.system_prompt + self.rubric + self.presentation_prompt + self.task_prompt
    
    def is_annotated(self) -> bool:
        return self.lower_bound is not None and self.upper_bound is not None
    
    def to_dict(self) -> Dict[str, Union[str, int, float, Dict, LLMResponse, None]]:
        return {
            "propensity": self.propensity,
            "system_prompt": self.system_prompt,
            "rubric": self.rubric,
            "task_prompt": self.task_prompt,
            "presentation_prompt": self.presentation_prompt,
            "source": self.source,
            "llm_response": self.llm_response.raw if isinstance(self.llm_response, LLMResponse) else self.llm_response,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "metadata": self.metadata
        }
    
    def _llm_call_single(
        self, 
        client: AzureOpenAI, 
        temperature: Optional[float],
        max_tokens: Optional[int],
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
    ):
        
        if schema == "free":
            output_structure = None
        else:
            if issubclass(schema, BaseModel):
                output_structure = azutils.pydantic_to_json_schema(model = schema)
            else:
                raise ValueError("If `schema` is provided, it must be a pydantic BaseModel subclass")

        full_prompt = self.get_full_prompt()

        deployment_model = self.source
        llm_response = azutils.llm_single_response(
            client = client,
            deployment_model = deployment_model,
            prompt = full_prompt,
            temperature = temperature,
            max_tokens = max_tokens,
            return_metadata = True,
            output_structure = output_structure
        )

        self.llm_response = llm_response #type: ignore
    
    def _parse_free_text_llm_output(
        self,
        pattern: re.Pattern | str = FINAL_RANGE_PATTERN,
        verbosity: int = 1
    ) -> None:

        """
        Extract the propensity range from a free-form LLM answer.

        Raises:
            ValueError:
                If the instance has no LLM response yet, or if the response
                does not contain a match for `pattern`.
        """

        # Parse schema output
        if self.llm_response is None:
            raise ValueError("This instance has not been annotated yet")

        matches = re.findall(pattern, self.llm_response.text) #type: ignore
        if verbosity > 2:
            print(matches)
        if not matches:
            raise ValueError("No FINAL_RANGE found in the LLM output")

        lb, ub = map(int, matches[-1])

        self.lower_bound = lb
        self.upper_bound = ub
        if self.metadata is None:
            self.metadata = {
                "explanation": self.llm_response.text #type: ignore
            }
        else:
            self.metadata.update(
                {"explanation": self.llm_response.text} #type: ignore
            )

    def _parse_structured_llm_output(
        self,
        schema: Type[BaseModel] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound"
    ) -> None:
        
        """
        Parse structured LLM output into the current PropensityAnnotation object.

        This method extracts the lower and upper bounds from the structured output
        and stores any additional fields into the `metadata` attribute.

        Args:
            llm_response (LLMResponse):
                Response object returned by the LLM call.

            schema (Type[BaseModel]):
                Pydantic model defining the expected structured output.

        Raises:
            ValueError:
                If parsing fails or required fields are missing.

        Notes:
            - Expects the LLM response to contain JSON-compatible structured output.
            - The schema must define `lower_bound` and `upper_bound` fields.
            - Any additional fields in the schema are stored in `metadata`.
        """

        if schema == "free":
            raise ValueError("Cannot parse free-form output with structured parser")

        # Parse schema output
        if self.llm_response is None:
            raise ValueError("This instance has not been annotated yet")

        try:
            parsed_response = json.loads(self.llm_response.text) #type: ignore
        except Exception as ex:
            raise ValueError(f"Failed to parse LLM output as JSON: {ex}")
        
        # Validate with schema
        try:
            parsed_obj = schema(**parsed_response)
        except Exception as ex:
            raise ValueError(f"Failed to validate LLM output with schema: {ex}")
        
        # Get lower and upper bounds
        if not hasattr(parsed_obj, lower_bound_field) or not hasattr(parsed_obj, upper_bound_field):
            raise ValueError("Schema must define `lower_bound_field` and `upper_bound_field` fields")

        self.lower_bound = getattr(parsed_obj, lower_bound_field)
        self.upper_bound = getattr(parsed_obj, upper_bound_field)

        # Store rest of params as metadata
        parsed_data = parsed_obj.model_dump()

        new_metadata = {
            k: v for k, v in parsed_data.items()
            if k not in {"lower_bound", "upper_bound"}
        }

        if new_metadata:
            if self.metadata is None:
                self.metadata = new_metadata
            else:
                self.metadata.update(new_metadata)

    def annotate(
        self,
        client: AzureOpenAI,
        annotator: Literal["model", "human"] = "model",
        annotator_temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound",
        regex: Optional[re.Pattern | str] = None,
        verbosity: int = 1
    ):

        """
        Annotate this instance with a single (synchronous) call to the LLM.

        Args:
            client (AzureOpenAI):
                An initialized Azure OpenAI client.

            annotator (Literal["model", "human"]):
                Only "model" is implemented.

            annotator_temperature (float | None):
                Sampling temperature. Pass None for reasoning deployments
                (o-series, gpt-5.x), which reject the parameter.

            max_tokens (int | None):
                Maximum number of output tokens. For reasoning deployments this
                budget also covers the (hidden) reasoning tokens.

            schema (Literal["free"] | Type[BaseModel]):
                "free" asks for free-form text, parsed with `regex`. A pydantic
                model asks for structured JSON output, parsed against the model.

            regex (re.Pattern | str | None):
                Pattern used to extract the range when `schema = "free"`.
                Defaults to `FINAL_RANGE_PATTERN`, i.e. the `<FINAL_RANGE>[LB, UB]</FINAL_RANGE>`
                tag requested by `rubrics/presentation.txt`.

        Raises:
            ValueError:
                If the LLM output cannot be parsed into a propensity range.
        """

        if annotator != "model":
            raise NotImplementedError("Only `annotator = 'model'` is implemented")

        # Call LLM annotator
        self._llm_call_single(
            client = client,
            temperature = annotator_temperature,
            max_tokens = max_tokens,
            schema = schema)

        if verbosity > 0:
            print("Request sent and responded by the annotator model")

        # Parse results
        if schema != "free":
            self._parse_structured_llm_output(
                schema = schema,
                lower_bound_field = lower_bound_field,
                upper_bound_field = upper_bound_field
            )

        else:
            self._parse_free_text_llm_output(
                pattern = regex if regex is not None else FINAL_RANGE_PATTERN,
                verbosity = verbosity
            )

        if verbosity > 0:
            print("Annotation successfully parsed")
            

@dataclass(slots = True)
class PropAnnotationCollection:
    annotations: List[PropensityAnnotation]
    batch_id: Optional[str] = None
    batch_requests: Optional[List] = None
    batch_filename: Optional[Union[str, Path]] = None
    output_filename: Optional[Union[str, Path]] = None
    error_filename: Optional[Union[str, Path]] = None

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        encoding: str = "utf-8"
    ) -> PropAnnotationCollection:

        annotations = []
        with open(path, "r", encoding = encoding) as f:
            for line in f:
                data = json.loads(line)
                annotations.append(PropensityAnnotation(**data))
        return cls(annotations = annotations)

    def __getitem__(self, index):
        return self.annotations[index]

    def __iter__(self):
        for ann in self.annotations:
            yield ann

    def to_list(self) -> List[Dict]:
        return [ann.to_dict() for ann in self.annotations]

    def save_jsonl(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8"
    ) -> None:
        
        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)

        with open(path, "w", encoding = encoding) as f:
            for ann in self.annotations:
                # `default = str` keeps raw API payloads (which may carry
                # non-JSON-native objects) from breaking the whole dump
                f.write(json.dumps(ann.to_dict(), default = str) + "\n")

    def save_csv(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        fields: Optional[List[str]] = None,
        *args,
        **kwargs
    ):
        DEFAULT_FIELDS = [
                "propensity",
                "system_prompt",
                "rubric",
                "presentation_prompt",
                "task_prompt",
                "source",
                "lower_bound",
                "upper_bound",
                "metadata"
            ]

        if fields is None:
            fields = DEFAULT_FIELDS

        # `old_fields` live on the annotation itself, `new_fields` are looked up
        # in its `metadata` dict. Both lists must exist for either branch.
        old_fields = [f for f in fields if f in DEFAULT_FIELDS]
        new_fields = [f for f in fields if f not in DEFAULT_FIELDS]

        info = {}
        for f in fields:
            info[f] = list()

        for ann in self.annotations:

            # Common already existing fields
            for f in old_fields:
                info[f].append(getattr(ann, f, None))

            # Check metadata for new fields
            metadata = ann.metadata if ann.metadata is not None else dict()
            for f in new_fields:
                info[f].append(metadata.get(f, None))

        df = pd.DataFrame(info)
        df = df.loc[:, fields]

        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)

        df.to_csv(path, *args, encoding = encoding, **kwargs)

        return df

    def _prepare_batch(
        self,
        batch_filename: Union[str, Path, None],
        temperature: float,
        max_tokens: Union[int, None],
        custom_ids: Optional[Union[List[str], Literal["metadata"]]] = None,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema
    ):
        
        if schema == "free":
            output_structure = None
        else:
            if issubclass(schema, BaseModel):
                output_structure = azutils.pydantic_to_json_schema(model = schema)
            else:
                raise ValueError("If `schema` is provided, it must be a pydantic BaseModel subclass")

        full_prompts = []
        cids = []
        models = set()

        for prop_annotation in self.annotations:
            full_prompt = prop_annotation.get_full_prompt()
            full_prompts.append(full_prompt)

            model = prop_annotation.source
            models.add(model)

            # Custom id is found in the metadata field of the annotation with the field name of "custom_id"
            if custom_ids == "metadata":
                if prop_annotation.metadata is None or "custom_id" not in prop_annotation.metadata:
                    raise ValueError(
                        "custom_ids = 'metadata' requires every annotation to carry "
                        "a 'custom_id' entry in its metadata dict"
                    )
                cids.append(prop_annotation.metadata["custom_id"])

        if len(models) != 1:
            raise ValueError("Collection of annotations must have the same annotator model")
            
        # Add custom_ids if not present in metadata
        if custom_ids is None:
            for i in range(len(full_prompts)):
                c_id = f"{i:>07}"
                cids.append(c_id)

                # Add to metadata as well
                if self.annotations[i].metadata is None:
                    self.annotations[i].metadata = {}

                self.annotations[i].metadata["custom_id"] = c_id #type: ignore

        elif custom_ids != "metadata":
            for i, c_id in enumerate(custom_ids):
                cids.append(c_id)   

                # Add to metadata as well
                if self.annotations[i].metadata is None:
                    self.annotations[i].metadata = {}

                self.annotations[i].metadata["custom_id"] = c_id #type: ignore

        assert len(cids) == len(full_prompts), "custom_ids must have the same number of elements as the number of annotations"

        data = [
            {"custom_id" : cids[i], "prompt": full_prompts[i]} for i in range(len(cids)) 
        ]

        requests = azutils.create_batch_requests(
            prompts = data,
            deployment_model = models.pop(),
            batch_filename = batch_filename,
            id_key = "custom_id",
            prompt_key = "prompt",
            temperature = temperature,
            max_tokens = max_tokens,
            output_structure = output_structure
        )

        self.batch_requests = requests
        self.batch_filename = batch_filename
    
    def _submit_batch(
        self,
        client: AzureOpenAI,
        encoding: str = "utf-8"
    ):
        
        batch_id = azutils.submit_batch(
            client = client,
            batch_requests = self.batch_requests,
            batch_input_path = self.batch_filename,
            encoding = encoding
        )

        self.batch_id = batch_id

    def _retrieve_batch_results(
        self,
        client: AzureOpenAI,
        output_path: Optional[Union[str, Path]] = None,
        error_path: Optional[Union[str, Path]] = None,
        parse_json: bool = True,
    ):
        
        if self.batch_id is None:
            raise ValueError("Please first submit the batch to get a batch_id")

        responses = azutils.retrieve_batch_results(
            client = client,
            batch_id = self.batch_id,
            output_path = output_path,
            error_path = error_path,
            parse_json = parse_json,
            return_format = "LLMResponse"
        )

        self.output_filename = output_path
        self.error_filename = error_path

        # Create temporary dict for efficient accesing
        temp_dict = {}
        for ann in self.annotations:
            c_id = ann.metadata["custom_id"] #type: ignore
            if c_id in temp_dict:
                raise KeyError("Two annotations have the same custom_id")
            else:
                temp_dict[c_id] = ann

        for c_id, llm_response in responses["outputs"].items():
            if c_id in temp_dict:
                temp_dict[c_id].llm_response = llm_response
            else:
                logging.warning(f"Received unknown custom_id: {c_id}")

    def _parse_free_text_llm_output(
        self,
        regex: re.Pattern | str = FINAL_RANGE_PATTERN,
        verbosity: int = 1
        ) -> Tuple[int, int, int]:

        total = len(self.annotations)
        success = 0
        errors = 0

        if verbosity > 1:
            for ann in tqdm(self.annotations, desc = "Parsing LLM responses"):
                try:
                    ann._parse_free_text_llm_output(
                        pattern = regex
                    )
                    success += 1
                except Exception as ex:
                    logging.warning(f"Failed to annotate annotation: {ex}")
                    errors += 1
                    continue

        else:
            for ann in self.annotations:
                try:
                    ann._parse_free_text_llm_output(
                        pattern = regex
                    )
                    success += 1
                except Exception as ex:
                    logging.warning(f"Failed to annotate annotation: {ex}")
                    errors += 1
                    continue

        return success, errors, total

    def _parse_structured_output(
        self,
        schema: Type[BaseModel] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound",
        verbosity: int = 1
    ) -> Tuple[int, int, int]:
        
        total = len(self.annotations)
        success = 0
        errors = 0

        if verbosity > 1:
            for ann in tqdm(self.annotations, desc = "Parsing LLM responses"):
                try:
                    ann._parse_structured_llm_output(
                        schema = schema,
                        lower_bound_field = lower_bound_field,
                        upper_bound_field = upper_bound_field
                    )
                    success += 1
                except ValueError as ex:
                    logging.warning(f"Failed to annotate annotation: {ex}")
                    errors += 1
                    continue

        else:
            for ann in self.annotations:
                try:
                    ann._parse_structured_llm_output(
                        schema = schema,
                        lower_bound_field = lower_bound_field,
                        upper_bound_field = upper_bound_field
                    )
                    success += 1
                except ValueError as ex:
                    logging.warning(f"Failed to annotate annotation: {ex}")
                    errors += 1
                    continue


        return success, errors, total

    def prepare_and_send_batch(
        self,
        client: AzureOpenAI,
        batch_filename: Union[str, Path, None],
        temperature: float,
        max_tokens: Union[int, None],
        custom_ids: Optional[Union[List[str], Literal["metadata"]]] = None,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
        encoding: str = "utf-8"
    ):
        
        self._prepare_batch(
            batch_filename = batch_filename,
            temperature = temperature,
            max_tokens = max_tokens,
            custom_ids = custom_ids,
            schema = schema
        )

        self._submit_batch(
            client = client,
            encoding = encoding
        )

    def retrieve_and_parse_batch(
        self,
        client: AzureOpenAI,
        output_path: Optional[Union[str, Path]] = None,
        error_path: Optional[Union[str, Path]] = None,
        parse_json: bool = True,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound",
    ):
        
        self._retrieve_batch_results(
            client = client,
            output_path = output_path,
            error_path = error_path,
            parse_json = parse_json
        )
        
        self._parse_structured_output(
            schema = schema, #type: ignore
            lower_bound_field = lower_bound_field,
            upper_bound_field = upper_bound_field
        )

    def annotate_batch(
        self,
        client: AzureOpenAI,
        annotator: Literal["model", "human"] = "model",
        batch_filename: Optional[Union[str, Path]] = None,
        custom_ids: Optional[Union[List[str], Literal["metadata"]]] = None,
        annotator_temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound",
        retry_time: int = 60,
        timeout: int = 3600,
        regex: Optional[re.Pattern | str] = None,
        verbosity: int = 1
    ):
        
        if annotator == "model":
            # Create batch and submit
            self._prepare_batch(
                batch_filename = batch_filename,
                temperature = annotator_temperature,
                max_tokens = max_tokens,
                custom_ids = custom_ids,
                schema = schema
            )

            self._submit_batch(
                client = client
            )

            if verbosity > 0:
                print(f"Annotation successfully sent in batch form, batch id: {self.batch_id}")

            # See if batch is complete every retry_time seconds:
            done = False
            start_time = time.time()
            while not done:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Batch {self.batch_id} timed out")
                status = client.batches.retrieve(self.batch_id).status #type: ignore
                if status == "completed":
                    self._retrieve_batch_results(
                        client = client,
                        output_path = self.output_filename,
                        error_path = self.error_filename
                    )
                    done = True
                elif status in {"cancelled", "failed"}:
                    raise RuntimeError(f"Batch {self.batch_id} ended with status: {status}")
                else:
                    if verbosity > 1:
                        print(f"Current status of annotation batch: {status}")
                    time.sleep(retry_time)

            if verbosity > 0:
                print("Batch successfully completed. Parsing results...")

            if schema != "free":
                # Parse results and fill annotation items
                success, errors, total = self._parse_structured_output(
                    schema = schema,
                    lower_bound_field = lower_bound_field,
                    upper_bound_field = upper_bound_field
                )
            
                if verbosity > 0:
                    print("Batch successfully parsed")
                if verbosity > 1:
                    print(f"{success}/{total} ({round(success/total*100, 2)}%) successes, {errors}/{total} ({round(errors/total*100, 2)}%) errors")

            else:
                if regex is None:
                    raise ValueError("regex should be a pattern, not None")
                success, errors, total = self._parse_free_text_llm_output(
                    regex, verbosity = verbosity
                )

        else:
            raise NotImplementedError
        

    def annotate_sequential(
        self,
        client: AzureOpenAI,
        annotator: Literal["model", "human"] = "model",
        annotator_temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound",
        regex: Optional[re.Pattern | str] = None,
        skip_annotated: bool = True,
        max_retries: int = 3,
        retry_time: float = 5.0,
        verbosity: int = 1
    ) -> Tuple[int, int, int]:

        """
        Annotate every instance in the collection with one LLM call each.

        Unlike `annotate_batch`, requests are sent one at a time, so results are
        available immediately and partial progress survives a failure. Instances
        that fail every retry are left unannotated and reported in the summary.

        Returns:
            tuple[int, int, int]:
                (successes, errors, total).
        """

        total = len(self.annotations)
        success = 0
        errors = 0

        for ann in tqdm(self.annotations, desc = "Annotating sequentially"):

            if skip_annotated and ann.is_annotated():
                success += 1
                continue

            last_error = None
            for attempt in range(max_retries):
                try:
                    ann.annotate(
                        client = client,
                        annotator = annotator,
                        annotator_temperature = annotator_temperature,
                        max_tokens = max_tokens,
                        schema = schema,
                        lower_bound_field = lower_bound_field,
                        upper_bound_field = upper_bound_field,
                        regex = regex,
                        verbosity = verbosity - 1
                    )
                    success += 1
                    last_error = None
                    break
                except Exception as ex:
                    last_error = ex
                    if attempt < max_retries - 1:
                        # Rate limits and transient 5xx are the common case here
                        time.sleep(retry_time * (attempt + 1))

            if last_error is not None:
                c_id = (ann.metadata or {}).get("custom_id", "<no custom_id>")
                logging.warning(f"Failed annotation {c_id}: {last_error}")
                errors += 1

        if verbosity > 0 and total > 0:
            print(
                f"{success}/{total} ({round(success/total*100, 2)}%) successes, "
                f"{errors}/{total} ({round(errors/total*100, 2)}%) errors"
            )

        return success, errors, total
