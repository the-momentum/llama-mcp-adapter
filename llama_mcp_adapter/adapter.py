from types import EllipsisType
from typing import Any, Dict, List, Literal, Union

import llama_index.tools.mcp.base as llama_mcp_base
from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import McpToolSpec
from pydantic import BaseModel, Field, create_model


class TypeResolutionMixin:
    def _resolve_field_type(
        self: "McpToolSpecAdapter",
        field_schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve the Python type from a field schema."""
        if "$ref" in field_schema:
            return self._resolve_reference(field_schema, defs)
        if "enum" in field_schema:
            return Literal[tuple(field_schema["enum"])]
        if "anyOf" in field_schema:
            return self._resolve_union_type(field_schema, defs)
        return self._resolve_basic_type(field_schema, defs)

    def _resolve_reference(
        self: "McpToolSpecAdapter",
        field_schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve a $ref reference."""
        ref_name = self._extract_ref_name(field_schema["$ref"])

        if ref_name not in defs:
            return self.properties_cache.get(ref_name) or self._create_model(
                defs[ref_name],
                ref_name,
                defs,
            )

        ref_schema = defs[ref_name]

        if "anyOf" in ref_schema:
            return self._resolve_union_type(ref_schema, defs)
        if self._is_simple_array(ref_schema):
            return self._create_list_type(ref_schema, defs)
        if self._is_simple_object(ref_schema):
            return self._create_dict_type(ref_schema, defs)
        return self.properties_cache.get(ref_name) or self._create_model(
            ref_schema,
            ref_name,
            defs,
        )

    def _resolve_union_type(
        self: "McpToolSpecAdapter",
        schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve a Union type (anyOf)."""
        union_types = [self._resolve_union_option(option, defs) for option in schema["anyOf"]]
        return Union[tuple(union_types)] if len(union_types) > 1 else union_types[0]

    def _resolve_union_option(
        self: "McpToolSpecAdapter",
        option: dict,
        defs: dict,
    ) -> Any:
        """Resolve a single option in a union type."""
        if "$ref" in option:
            return self._resolve_reference(option, defs)
        if option.get("type") == "null":
            return type(None)
        return self._resolve_basic_type(option, defs)

    def _resolve_basic_type(
        self: "McpToolSpecAdapter",
        schema: dict,
        defs: dict,
    ) -> Any:
        """Resolve a basic JSON Schema type."""
        json_type = schema.get("type", "string")
        json_type = json_type[0] if isinstance(json_type, list) else json_type

        if self._is_simple_array(schema):
            return self._create_list_type(schema, defs)
        if self._is_simple_object(schema):
            return self._create_dict_type(schema, defs)
        return self.json_type_mappings.get(json_type, str)


class TypeCreationMixin:
    def _create_list_type(self: "McpToolSpecAdapter", schema: dict, defs: dict) -> type:
        """Create a List type from schema."""
        item_type = self._resolve_field_type(schema["items"], defs)
        return List[item_type]

    def _create_dict_type(self: "McpToolSpecAdapter", schema: dict, defs: dict) -> type:
        """Create a Dict type from schema."""
        value_type = self._resolve_field_type(schema["additionalProperties"], defs)
        return Dict[str, value_type]

    def _is_simple_array(self: "McpToolSpecAdapter", schema: dict) -> bool:
        """Check if schema is a simple array type."""
        return schema.get("type") == "array" and "items" in schema

    def _is_simple_object(self: "McpToolSpecAdapter", schema: dict) -> bool:
        """Check if schema is a simple object type."""
        return schema.get("type") == "object" and "additionalProperties" in schema

    def _extract_ref_name(self: "McpToolSpecAdapter", ref_path: str) -> str:
        """Extract reference name from $ref path."""
        return ref_path.split("#/$defs/")[-1]


class FieldExtractionMixin:
    def _extract_fields(self: "McpToolSpecAdapter", schema: dict, defs: dict) -> dict:
        """Extract Pydantic fields from schema."""
        properties = self._get_properties(schema)
        required_fields = set(schema.get("required", []))

        # For enum schemas, treat them as required by default
        if "enum" in schema:
            required_fields = {schema.get("title", "enum_field")}

        print(f"\n❗required_fields: {required_fields}")

        fields = {}
        for field_name, field_schema in properties.items():
            field_type = self._resolve_field_type(field_schema, defs)
            default_value, final_type = self._set_field_default(
                field_name,
                required_fields,
                field_type,
                field_schema,
            )

            fields[field_name] = (
                final_type,
                Field(default_value, description=field_schema.get("description", "")),
            )

        return fields

    def _get_properties(self: "McpToolSpecAdapter", schema: dict) -> dict:
        """Get properties from schema, handling enum types."""
        if "enum" in schema:
            # For enum types, create a property with the schema name as the key
            # This ensures the enum is treated as a required field
            return {schema.get("title", "enum_field"): schema}
        return schema.get("properties", {})

    @staticmethod
    def _set_field_default(
        field: str,
        required_fields: set[str],
        ftype: Any,
        field_schema: dict,
    ) -> tuple[EllipsisType | None, Any]:
        """Set default value and make type optional if needed."""
        if field in required_fields:
            return ..., ftype
        default_value = field_schema.get("default")
        if default_value is None:
            ftype = ftype | type(None)
        return default_value, ftype


class McpToolSpecAdapter(
    TypeResolutionMixin,
    TypeCreationMixin,
    FieldExtractionMixin,
    McpToolSpec,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.properties_cache = {}
        self.json_type_mappings = {
            **llama_mcp_base.json_type_mapping,
            "array": List,
            "object": Dict,
        }

    async def to_tool_list_async(self) -> list[FunctionTool]:
        """Override original function to use custom model creation logic."""
        parent_ptr = llama_mcp_base.create_model_from_json_schema
        llama_mcp_base.create_model_from_json_schema = self.create_model_from_json_schema

        try:
            return await super().to_tool_list_async()
        finally:
            llama_mcp_base.create_model_from_json_schema = parent_ptr

    def create_model_from_json_schema(
        self,
        schema: dict[str, Any],
        model_name: str = "DynamicModel",
    ) -> type[BaseModel]:
        """Create Pydantic model from JSON schema with custom type handling."""
        defs = schema.get("$defs", {})

        # Process all type definitions
        for cls_name, cls_schema in defs.items():
            self.properties_cache[cls_name] = self._create_model(
                cls_schema,
                cls_name,
                defs,
            )

        return self._create_model(schema, model_name)

    def _create_model(
        self,
        schema: dict,
        model_name: str,
        defs: dict = {},
    ) -> type[BaseModel]:
        """Create a Pydantic model from a schema."""
        if model_name in self.properties_cache:
            return self.properties_cache[model_name]

        fields = self._extract_fields(schema, defs)
        model = create_model(model_name, **fields)
        self.properties_cache[model_name] = model
        return model
