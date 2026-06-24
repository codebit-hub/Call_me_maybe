# Parser: Imports, validates, instantiates JSON file

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterSpec:
    """Defines the expected data type of a single function parameter"""

    type_name: str


@dataclass(frozen=True)
class FunctionSpec:
    """Defines a callable target function and its params"""

    name: str
    description: str
    parameters: dict[str, ParameterSpec]

    @classmethod
    def parse(cls, raw_data: dict[str, Any]) -> "FunctionSpec":
        """Constructs FunctionSpec from a raw JSON dict item"""

        raw_params: dict[str, dict[str, Any]] = raw_data.get("parameters", {})
        parsed_params: dict[str, ParameterSpec] = {}

        for param_name, param_info in raw_params.items():
            parsed_params[param_name] = ParameterSpec(
                type_name=str(param_info.get("type", "string"))
            )

        return cls(
            name=str(raw_data.get("name", "")),
            description=str(raw_data.get("description", "")),
            parameters=parsed_params
        )


# Receives file path, opens file, loads json file into raw_list,
# Uses FunctionSpec.parse() to get parameters dict and build
# registry dicts[str, params]
def load_schema(filepath: str) -> dict[str, FunctionSpec]:
    """Loads and indexes function definitions from JSON file"""

    with open(filepath, "r", encoding="utf-8") as f:
        raw_list: list[dict[str, Any]] = json.load(f)

    registry: dict[str, FunctionSpec] = {}
    for item in raw_list:
        spec = FunctionSpec.parse(item)
        if spec.name:
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
                print(f"    --{p_name}: <{p_spec.type_name}>")

    except Exception as err:
        print(f"\n[ERROR]: Failed to parse schema: {err}")
        sys.exit(1)
