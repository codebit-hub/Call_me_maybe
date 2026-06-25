# Parser: Imports, validates, instantiates JSON file

import json
from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    """The expected data type of a single function parameter"""

    type: str


class ReturnSpec(BaseModel):
    """The expected return type of the function"""

    type: str


class FunctionSpec(BaseModel):
    """Defines a callable target function and its params"""

    name: str
    description: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec | None = None


def load_schema(filepath: str) -> dict[str, FunctionSpec]:
    """
    Loads, validates and indexes function definitions from a JSON file.
    Returns a dictionary mapping: function_name -> FunctionSpec.
    """

    # raw_list contains the same content as in json file:
    # [{'name': 'fn_add_numbers', 'description': 'Add numbers...',
    # 'parameters': {'a': {'type': 'number'}, 'b': {'type': 'number'}},
    # 'returns': {'type': 'number'}},
    #  and so on.
    with open(filepath, "r", encoding="utf-8") as f:
        raw_list: list[dict[str, object]] = json.load(f)

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

    for item in raw_list:
        spec = FunctionSpec.model_validate(item)
        registry[spec.name] = spec
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
