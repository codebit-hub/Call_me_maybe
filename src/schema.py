# Parser: Imports, validates, instantiates JSON file

import json
from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    """Defines the expected data type of a single function parameter"""

    type: str


class ReturnSpec(BaseModel):
    """Defines the expected return type of the function."""

    type: str


class FunctionSpec(BaseModel):
    """Defines a callable target function and its params"""

    name: str
    description: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec | None = None


def load_schema(filepath: str) -> dict[str, FunctionSpec]:
    """
    Loads, validates, and indexes function definitions from a JSON file.
    Returns a dictionary mapping: function_name -> FunctionSpec
    """

    with open(filepath, "r", encoding="utf-8") as f:
        raw_list: list[dict[str, object]] = json.load(f)

    registry: dict[str, FunctionSpec] = {}
    for item in raw_list:
        spec = FunctionSpec.model_validate(item)
        registry[spec.name] = spec
    return registry


if __name__ == "__main__":
    import sys

    test_path = "data/input/functions_definition.json"
    print(f"Attempting to ingest schema from: {test_path}...")

    try:
        funcs = load_schema(test_path)
        print(f"[SUCCESS]: Inexed {len(funcs)} functions.\n")

        for f_name, f_spec in funcs.items():
            print(f"--{f_name} ({len(f_spec.parameters)} params)")
            for p_name, p_spec in f_spec.parameters.items():
                print(f"    --{p_name}: <{p_spec.type}>")

    except Exception as err:
        print(f"\n[ERROR]: Failed to parse schema: {err}")
        sys.exit(1)
