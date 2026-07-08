# Parser: Imports, validates, instantiates functions_definition.json

import json
import os
from pydantic import BaseModel, Field, ValidationError


class ParameterSpec(BaseModel):
    """The expected data type of a single function parameter"""

    # pattern=r'\S' ensures str contains at least one non-whitespace char
    # pattern= defines a regex constraint the string must match
    # r'\S' a raw string regex; r'' raw string,
    # \S - any non-whitespace character

    type: str = Field(pattern=r'\S')


class ReturnSpec(BaseModel):
    """The expected return type of the function"""

    type: str = Field(pattern=r'\S')


class FunctionSpec(BaseModel):
    """Defines a callable target function and its params"""

    name: str = Field(pattern=r'\S')
    description: str = Field(pattern=r'\S')
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec | None


def load_schema(filepath: str) -> dict[str, FunctionSpec]:
    """
    Loads, validates and indexes function definitions from a JSON file.
    Returns a dictionary mapping: function_name -> FunctionSpec.
    """
    # Validation

    # Handle completely empty files
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Schema file is completely empty: {filepath}")

    # raw_list contains the same content as in json file:
    # [{'name': 'fn_add_numbers', 'description': 'Add numbers...',
    # 'parameters': {'a': {'type': 'number'}, 'b': {'type': 'number'}},
    # 'returns': {'type': 'number'}},
    #  and so on.

    # Handle malformed JSON or files containing just spaces
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_list: list[dict[str, object]] = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {filepath}: {e}")

    # Handle valid JSON that is the wrong data type (e.g., a single string "")
    if not isinstance(raw_list, list):
        raise ValueError(
            f"Invalid schema format. Expected a list of functions, "
            f"but got {type(raw_list).__name__}."
        )

    # Handle an empty list []
    if not raw_list:
        raise ValueError("Schema list cannot be empty.")

    # Buils function data: FunctionSpec and sets it in the registry
    # item = spec:
    # name='fn_add_numbers'
    # description='Add two numbers together and return their sum.'
    # parameters={'a': ParameterSpec(type='number'),
    #             'b': ParameterSpec(type='number')}
    # returns=ReturnSpec(type='number')

    registry: dict[str, FunctionSpec] = {}

    # For each entry in JSON, turned in a validated object spec
    # spec.name = function name
    # spec.FunctionSpec = full function contents

    # Strictly validate missing keys and types using Pydantic
    for i, item in enumerate(raw_list):
        try:
            spec = FunctionSpec.model_validate(item)
            registry[spec.name] = spec

        except ValidationError as e:
            raise ValueError(
                f"Validation failed for function at index {i}:\n{e}"
                )
    return registry

    # registry:
    # {'fn_add_numbers': FunctionSpec(name='fn_add_numbers',
    # description='Add two numbers together and return their sum.',
    # parameters={'a': ParameterSpec(type='number'),
    #             'b': ParameterSpec(type='number')},
    # returns=ReturnSpec(type='number'))}


if __name__ == "__main__":
    import sys

    # Define the path to JOSON file
    test_path = "data/input/functions_definition.json"
    print(f"Attempting to ingest schema from: {test_path}...")

    try:
        # Parse, validate and record JSON file contents into registry
        funcs: dict[str, FunctionSpec] = load_schema(test_path)

        print(f"[SUCCESS]: Indexed {len(funcs)} functions.\n")

        for f_name, f_spec in funcs.items():
            print(f"--{f_name} ({len(f_spec.parameters)} params)")
            for p_name, p_spec in f_spec.parameters.items():
                print(f"    --{p_name}: <{p_spec.type}>")

    except Exception as err:
        print(f"\n[ERROR]: Failed to parse schema: {err}")
        sys.exit(1)
