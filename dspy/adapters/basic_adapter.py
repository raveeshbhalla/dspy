import json
import logging
import re
from typing import Any, Literal, get_origin

import json_repair
import litellm
import pydantic
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    serialize_for_json,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


def _is_simple_type(annotation) -> bool:
    """Check if annotation is a simple scalar type that doesn't need structured output."""
    return annotation in (str, int, float, bool)


def _has_open_ended_mapping(signature: SignatureMeta) -> bool:
    """Check if any output field has an open-ended mapping type like dict[str, Any]."""
    for field in signature.output_fields.values():
        if get_origin(field.annotation) is dict:
            return True
    return False


def _get_structured_outputs_response_format(
    signature: SignatureMeta,
    use_native_function_calling: bool = True,
) -> type[pydantic.BaseModel]:
    """
    Builds a Pydantic model from a DSPy signature's output_fields for structured outputs.
    """
    for name, field in signature.output_fields.items():
        if get_origin(field.annotation) is dict:
            raise ValueError(
                f"Field '{name}' has an open-ended mapping type which is not supported by Structured Outputs."
            )

    fields = {}
    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if use_native_function_calling and annotation == ToolCalls:
            continue
        default = field.default if hasattr(field, "default") else ...
        fields[name] = (annotation, default)

    pydantic_model = pydantic.create_model(
        "DSPyProgramOutputs",
        __config__=pydantic.ConfigDict(extra="forbid"),
        **fields,
    )

    schema = pydantic_model.model_json_schema()

    for prop in schema.get("properties", {}).values():
        prop.pop("json_schema_extra", None)

    def enforce_required(schema_part: dict):
        if schema_part.get("type") == "object":
            props = schema_part.get("properties")
            if props is not None:
                schema_part["required"] = list(props.keys())
                schema_part["additionalProperties"] = False
                for sub_schema in props.values():
                    if isinstance(sub_schema, dict):
                        enforce_required(sub_schema)
            else:
                schema_part["properties"] = {}
                schema_part["required"] = []
                schema_part["additionalProperties"] = False
        if schema_part.get("type") == "array" and isinstance(schema_part.get("items"), dict):
            enforce_required(schema_part["items"])
        for key in ("$defs", "definitions"):
            if key in schema_part:
                for def_schema in schema_part[key].values():
                    enforce_required(def_schema)

    enforce_required(schema)
    pydantic_model.model_json_schema = lambda *args, **kwargs: schema

    return pydantic_model


class BasicAdapter(Adapter):
    """A simple adapter that formats messages naturally without DSPy-specific markers.

    The BasicAdapter sends messages in a clean format similar to how you'd use the Vercel AI SDK
    or other standard LLM interfaces. It avoids special markers like `[[ ## field ## ]]` and instead
    uses natural formatting with optional XML or markdown structure.

    Key features:
        - Natural message formatting without DSPy-specific markers
        - XML (default) or markdown formatting for structure
        - Plain text output for simple single-field responses (str, int, float, bool)
        - Native structured outputs for complex/multi-field responses
        - In-context examples section rather than user/assistant message pairs
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = True,
        format_type: Literal["xml", "markdown"] = "xml",
    ):
        """
        Args:
            callbacks: List of callback functions to execute during adapter methods.
            use_native_function_calling: Whether to enable native function calling capabilities.
            format_type: The formatting style to use for system messages. Either "xml" (default) or "markdown".
        """
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
        )
        self.format_type = format_type

    def _should_use_structured_output(self, signature: type[Signature]) -> bool:
        """Determine if structured output should be used based on signature output fields."""
        output_fields = signature.output_fields

        # Single output field with simple type -> no structured output needed
        if len(output_fields) == 1:
            field = list(output_fields.values())[0]
            if _is_simple_type(field.annotation):
                return False

        # Multiple fields or complex types -> use structured output
        return True

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)

        # Determine if we need structured output
        if self._should_use_structured_output(processed_signature):
            self._configure_structured_output(lm, lm_kwargs, processed_signature)

        formatted_inputs = self.format(processed_signature, demos, inputs)
        outputs = lm(messages=formatted_inputs, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs, lm, lm_kwargs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)

        if self._should_use_structured_output(processed_signature):
            self._configure_structured_output(lm, lm_kwargs, processed_signature)

        formatted_inputs = self.format(processed_signature, demos, inputs)
        outputs = await lm.acall(messages=formatted_inputs, **lm_kwargs)
        return self._call_postprocess(processed_signature, signature, outputs, lm, lm_kwargs)

    def _configure_structured_output(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
    ) -> None:
        """Configure lm_kwargs for structured output if supported."""
        provider = lm.model.split("/", 1)[0] or "openai"
        params = litellm.get_supported_openai_params(model=lm.model, custom_llm_provider=provider)

        if not params or "response_format" not in params:
            return

        has_tool_calls = any(field.annotation == ToolCalls for field in signature.output_fields.values())
        supports_structured_outputs = litellm.supports_response_schema(model=lm.model, custom_llm_provider=provider)

        if _has_open_ended_mapping(signature) or (not self.use_native_function_calling and has_tool_calls):
            lm_kwargs["response_format"] = {"type": "json_object"}
            return

        if not supports_structured_outputs:
            lm_kwargs["response_format"] = {"type": "json_object"}
            return

        try:
            structured_output_model = _get_structured_outputs_response_format(
                signature, self.use_native_function_calling
            )
            lm_kwargs["response_format"] = structured_output_model
        except Exception:
            logger.warning("Failed to create structured output format, falling back to JSON mode.")
            lm_kwargs["response_format"] = {"type": "json_object"}

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format output field descriptions for the system message."""
        return f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the expected output structure description."""
        output_fields = signature.output_fields

        if not self._should_use_structured_output(signature):
            # Single simple output - just describe expected format
            field_name, field_info = list(output_fields.items())[0]
            type_name = get_annotation_name(field_info.annotation)
            return f"Respond with just the {field_name} value as plain {type_name}."

        # Multiple/complex outputs - describe JSON structure
        field_descriptions = []
        for name, field in output_fields.items():
            type_name = get_annotation_name(field.annotation)
            field_descriptions.append(f'  "{name}": <{type_name}>')

        fields_str = ",\n".join(field_descriptions)
        return f"Respond with a JSON object in this format:\n{{\n{fields_str}\n}}"

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description from signature instructions."""
        return signature.instructions.strip()

    def format_system_message(self, signature: type[Signature]) -> str:
        """Format the complete system message."""
        parts = []

        # Task description first
        task_desc = self.format_task_description(signature)
        if task_desc:
            parts.append(task_desc)

        # Output field descriptions
        parts.append(self.format_field_description(signature))

        # Output structure
        parts.append(self.format_field_structure(signature))

        return "\n\n".join(parts)

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format user message content.

        - Single input: Just the raw value
        - Multiple inputs: XML tags with field names (e.g., <context>...</context>)
        """
        input_fields = signature.input_fields
        provided_inputs = {k: v for k, v in inputs.items() if k in input_fields}

        parts = []
        if prefix:
            parts.append(prefix)

        if len(provided_inputs) == 1:
            # Single input - just the raw value
            value = list(provided_inputs.values())[0]
            field_info = list(input_fields.values())[0]
            formatted_value = format_field_value(field_info=field_info, value=value)
            parts.append(formatted_value)
        else:
            # Multiple inputs - use XML tags with field names
            for field_name, value in provided_inputs.items():
                field_info = input_fields[field_name]
                formatted_value = format_field_value(field_info=field_info, value=value)
                if self.format_type == "xml":
                    parts.append(f"<{field_name}>{formatted_value}</{field_name}>")
                else:
                    parts.append(f"## {field_name}\n{formatted_value}")

        if suffix:
            parts.append(suffix)

        return "\n\n".join(parts).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format assistant message content for examples."""
        output_fields = signature.output_fields

        if not self._should_use_structured_output(signature):
            # Single simple output - just the raw value
            field_name = list(output_fields.keys())[0]
            value = outputs.get(field_name, missing_field_message)
            if value is None:
                value = missing_field_message or ""
            return str(value)

        # Multiple/complex outputs - JSON format
        output_dict = {}
        for field_name in output_fields:
            value = outputs.get(field_name, missing_field_message)
            output_dict[field_name] = value

        return json.dumps(serialize_for_json(output_dict), indent=2)

    def format_demos(self, signature: type[Signature], demos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format few-shot examples as in-context examples in the system message.

        Unlike ChatAdapter which uses user/assistant message pairs, BasicAdapter
        includes examples directly in the system message.
        """
        # BasicAdapter includes demos in the system message, not as separate messages
        # This method returns empty list - demos are handled in format()
        return []

    def _format_demos_section(self, signature: type[Signature], demos: list[dict[str, Any]]) -> str:
        """Format demos as an examples section for the system message."""
        if not demos:
            return ""

        complete_demos = []
        for demo in demos:
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)
            if has_input and has_output:
                complete_demos.append(demo)

        if not complete_demos:
            return ""

        if self.format_type == "xml":
            examples_parts = ["<examples>"]
            for demo in complete_demos:
                example_parts = ["<example>"]

                # Format inputs
                for field_name in signature.input_fields:
                    if field_name in demo:
                        value = demo[field_name]
                        field_info = signature.input_fields[field_name]
                        formatted_value = format_field_value(field_info=field_info, value=value)
                        example_parts.append(f"<{field_name}>{formatted_value}</{field_name}>")

                # Format outputs
                if not self._should_use_structured_output(signature):
                    # Single simple output
                    field_name = list(signature.output_fields.keys())[0]
                    if field_name in demo:
                        value = demo[field_name]
                        example_parts.append(f"<output>{value}</output>")
                else:
                    # Multiple/complex outputs - JSON in output tag
                    output_dict = {}
                    for field_name in signature.output_fields:
                        if field_name in demo:
                            output_dict[field_name] = demo[field_name]
                    if output_dict:
                        example_parts.append(f"<output>{json.dumps(serialize_for_json(output_dict), indent=2)}</output>")

                example_parts.append("</example>")
                examples_parts.append("\n".join(example_parts))

            examples_parts.append("</examples>")
            return "\n".join(examples_parts)
        else:
            # Markdown format
            examples_parts = ["## Examples"]
            for i, demo in enumerate(complete_demos, 1):
                example_parts = [f"### Example {i}"]
                example_parts.append("**Input:**")

                for field_name in signature.input_fields:
                    if field_name in demo:
                        value = demo[field_name]
                        field_info = signature.input_fields[field_name]
                        formatted_value = format_field_value(field_info=field_info, value=value)
                        example_parts.append(f"- {field_name}: {formatted_value}")

                example_parts.append("**Output:**")
                if not self._should_use_structured_output(signature):
                    field_name = list(signature.output_fields.keys())[0]
                    if field_name in demo:
                        example_parts.append(str(demo[field_name]))
                else:
                    output_dict = {}
                    for field_name in signature.output_fields:
                        if field_name in demo:
                            output_dict[field_name] = demo[field_name]
                    if output_dict:
                        example_parts.append(f"```json\n{json.dumps(serialize_for_json(output_dict), indent=2)}\n```")

                examples_parts.append("\n".join(example_parts))

            return "\n\n".join(examples_parts)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format messages for the LM call."""
        from dspy.adapters.types.base_type import split_message_content_for_custom_types

        inputs_copy = dict(inputs)

        # Handle conversation history
        history_field_name = self._get_history_field_name(signature)
        if history_field_name:
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                inputs_copy,
            )

        messages = []

        # Build system message with demos included
        system_parts = [self.format_system_message(signature)]
        demos_section = self._format_demos_section(signature, demos)
        if demos_section:
            system_parts.append(demos_section)

        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # Add conversation history if present
        if history_field_name:
            messages.extend(conversation_history)
            content = self.format_user_message_content(signature_without_history, inputs_copy, main_request=True)
        else:
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)

        messages.append({"role": "user", "content": content})

        messages = split_message_content_for_custom_types(messages)
        return messages

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse LM output into a dictionary of output fields."""
        output_fields = signature.output_fields

        if not self._should_use_structured_output(signature):
            # Single simple output - parse directly
            field_name, field_info = list(output_fields.items())[0]
            try:
                value = parse_value(completion.strip(), field_info.annotation)
                return {field_name: value}
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="BasicAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse simple output: {e}",
                )

        # Multiple/complex outputs - parse as JSON
        # Try to extract JSON object from response
        pattern = r"\{(?:[^{}]|(?R))*\}"
        import regex

        match = regex.search(pattern, completion, regex.DOTALL)
        if match:
            completion = match.group(0)

        try:
            fields = json_repair.loads(completion)
        except Exception as e:
            raise AdapterParseError(
                adapter_name="BasicAdapter",
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse JSON response: {e}",
            )

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="BasicAdapter",
                signature=signature,
                lm_response=completion,
                message="LM response is not a JSON object.",
            )

        # Filter to only expected fields and parse values
        fields = {k: v for k, v in fields.items() if k in output_fields}

        for k, v in fields.items():
            if k in output_fields:
                fields[k] = parse_value(v, output_fields[k].annotation)

        if fields.keys() != output_fields.keys():
            raise AdapterParseError(
                adapter_name="BasicAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields
