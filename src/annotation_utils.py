from __future__ import annotations
from typing import Optional, List, Dict, Union, Type, Literal

import os
import re
import time
import json
import logging

from pathlib import Path
from pydantic import BaseModel
from dataclasses import dataclass

from openai import AzureOpenAI
import azure_utils as azutils
from azure_utils import LLMResponse

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
    llm_response: Optional[LLMResponse] = None
    lower_bound: Optional[int | float] = None
    upper_bound: Optional[int | float] = None
    metadata: Optional[Dict] = None

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
            "raw_response": self.llm_response,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "metadata": self.metadata
        }
    
    def _llm_call_single(
        self, 
        client: AzureOpenAI, 
        temperature: float,
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
            parsed_response = json.loads(self.llm_response.text)
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
        annotator_temeperature: float = 0.0,
        max_tokens: Optional[int] = None,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound"
    ):

        if annotator == "model":
            # Call LLM annotator
            self._llm_call_single(
                client = client,
                temperature = annotator_temeperature,
                max_tokens = max_tokens,
                schema = schema)
            
            # Parse results
            if schema != "free":
                self._parse_structured_llm_output(
                    schema = schema,
                    lower_bound_field = lower_bound_field,
                    upper_bound_field = upper_bound_field
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            

@dataclass(slots = True)
class PropAnnotationCollection:
    annotations: List[PropensityAnnotation]
    batch_id: Optional[str] = None
    batch_requests: Optional[List] = None
    batch_filename: Optional[Union[str, Path]] = None
    output_filename: Optional[Union[str, Path]] = None
    error_filename: Optional[Union[str, Path]] = None

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

            if custom_ids == "metadata":
                if prop_annotation.metadata is None:
                    prop_annotation.metadata = {}
            # Custom id is found in the metadata field of the annotation with the field name of "custom_id"
                cids.append(prop_annotation.metadata["custom_id"]) #type: ignore

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

    def _parse_structured_output(
        self,
        schema: Type[BaseModel] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound"
    ) -> None:
        
        for ann in self.annotations:
            try:
                ann._parse_structured_llm_output(
                    schema = schema,
                    lower_bound_field = lower_bound_field,
                    upper_bound_field = upper_bound_field
                )
            except ValueError as ex:
                logging.warning(f"Failed to annotate annotation: {ex}")
                continue

    def annotate(
        self,
        client: AzureOpenAI,
        annotator: Literal["model", "human"] = "model",
        batch_filename: Optional[Union[str, Path]] = None,
        custom_ids: Optional[Union[List[str], Literal["metadata"]]] = None,
        annotator_temeperature: float = 0.0,
        max_tokens: Optional[int] = None,
        schema: Union[Literal["free"], Type[BaseModel]] = PropAnnotationSchema,
        lower_bound_field: str = "lower_bound",
        upper_bound_field: str = "upper_bound",
        retry_time: int = 60,
        timeout: int = 3600
    ):
        
        if annotator == "model":
            # Create batch and submit
            self._prepare_batch(
                batch_filename = batch_filename,
                temperature = annotator_temeperature,
                max_tokens = max_tokens,
                custom_ids = custom_ids,
                schema = schema
            )

            self._submit_batch(
                client = client
            )

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
                    time.sleep(retry_time)

            if schema != "free":
                # Parse results and fill annotation items
                self._parse_structured_output(
                    schema = schema,
                    lower_bound_field = lower_bound_field,
                    upper_bound_field = upper_bound_field
                )
            
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError