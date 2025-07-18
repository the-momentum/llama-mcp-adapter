from types import EllipsisType
from typing import Any, Literal

from llama_index.tools.mcp import McpToolSpec
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field, create_model
import llama_index.tools.mcp.base as llama_mcp_base


class McpToolSpecAdapter(McpToolSpec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.properties_cache = {}
        self.json_type_mappings = llama_mcp_base.json_type_mapping

    async def to_tool_list_async(self) -> list[FunctionTool]:
        """
        Overwrite original function to use custom model creation logic.
        """
        parent_ptr = llama_mcp_base.create_model_from_json_schema
        llama_mcp_base.create_model_from_json_schema = (
            self.create_model_from_json_schema
        )

        try:
            return await super().to_tool_list_async()
        finally:
            llama_mcp_base.create_model_from_json_schema = parent_ptr

    def create_model_from_json_schema(
        self, schema: dict[str, Any], model_name: str = "DynamicModel"
    ) -> type[BaseModel]:
        """
        Adapter function to include Pydantic and other custom class parameters in schema.
        """
        custom_cls_defs: dict[str, Any] = schema.get("$defs", {})

        for cls_name, cls_schema in custom_cls_defs.items():
            # search only for clcustom objects like BaseModel or Literal
            self.properties_cache[cls_name] = self.add_property(
                cls_schema, cls_name, custom_cls_defs
            )

        # search for simple properties
        return self.add_property(schema, model_name)

    def add_property(
        self, property_schema: dict, property_name: str, defs: dict = {}
    ) -> type[BaseModel]:
        if property_name in self.properties_cache:
            return self.properties_cache[property_name]

        if "enum" in property_schema:
            properties = {property_name: property_schema}
        else:
            properties = property_schema.get("properties", {})

        required_fields = set(property_schema.get("required", []))
        fields = {}
        breakpoint()
        for field_name, field_schema in properties.items():
            if "$ref" in field_schema:
                ref_name = field_schema["$ref"].split("#/$defs/")[-1]
                field_type = self.properties_cache.get(ref_name) or self.add_property(
                    defs[ref_name], ref_name
                )
            elif "enum" in field_schema:
                field_type = Literal[tuple(field_schema["enum"])]
            else:
                json_type = field_schema.get("type", "string")
                json_type = json_type[0] if isinstance(json_type, list) else json_type
                field_type = llama_mcp_base.json_type_mapping.get(json_type, str)

            default_value, field_type = McpToolSpecAdapter.set_default(
                field_name, required_fields, field_type
            )

            fields[field_name] = (
                field_type,
                Field(default_value, description=field_schema.get("description", "")),
            )
        submodel = create_model(property_name, **fields)
        self.properties_cache[property_name] = submodel
        return submodel

    @staticmethod
    def set_default(
        field: str, required_fields: set[str], ftype: Any
    ) -> tuple[EllipsisType | None, Any]:
        if field in required_fields:
            default_value = ...
        else:
            default_value = None
            ftype = ftype | type(None)

        return default_value, ftype
