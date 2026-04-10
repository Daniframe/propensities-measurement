from __future__ import annotations
from typing import Optional, List, Dict, Union, Type, Literal

import os
import re
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

@dataclass
class PropensityAnnotation:
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

