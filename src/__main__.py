import json
import os
import sys
import time
import argparse
from typing import Union

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.schema import load_schema
from src.vocab import VocabMasks
from src.fsm_off import UnconstrainedGenerator
from src.fsm import JSONStateMachine


def main() -> None:
    """
    The main program
    """

    # CLI Arguments Parsing ======================================

    parser = argparse.ArgumentParser(
        prog="Constrained decoder",
        description="Constrained JSON LLM Generator")
    parser.add_argument(
        "-functions_definition", "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON schema defining available functions."
    )
    parser.add_argument(
        "-input", "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing test prompts."
    )
    parser.add_argument(
        "-output", "--output",
        type=str,
        default="data/output/function_calling_result.json",
        help="Path where the final JSON results will be saved."
    )
    parser.add_argument(
        "-off", "--off",
        action="store_true",
        help="Disable FSM constrained decoding and run raw LLM."
    )

    # Run the parser, place extracted data in
    # an argparse.Namespace object

    args = parser.parse_args()

    schema_path = args.functions_definition
    test_path = args.input
    output_path = args.output

    if not os.path.exists(schema_path):
        print(f"\n[ERROR]: Schema file not found at {schema_path}")
        sys.exit(1)

    if not os.path.exists(test_path):
        print(f"\n[ERROR]: Test prompts file not found at {test_path}")
        sys.exit(1)

    # Validates/Creates the output path based on user input
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Boot Up the Environment =================================
    print("=== Booting Environment ===")

    registry = load_schema(schema_path)
    print(f"+ Loaded {len(registry)} functions from schema.")

    model = Small_LLM_Model(device="cpu")
    print("+ LLM has been deployed.")

    # Swap FSM mode based on --off flag

    generator: Union[UnconstrainedGenerator, JSONStateMachine]
    if args.off:
        print("+ FSM is OFF. Constrained decoding deactivated. Raw LLM only!")
        generator = UnconstrainedGenerator(model, registry)
    else:
        masks = VocabMasks(model)
        print("+ Vocabulary masks have been aggrigated and set.")
        generator = JSONStateMachine(model, registry, masks)
        print("+ FSM has been deployed and is ready.")

    # Load the test cases ======================================
    print("\n=== Processing Prompts ===")
    print(f"+ Loading test prompts from {test_path}.")

    with open(test_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"+ Loaded {len(test_cases)} tests to process.")

    t_test_start = time.time()
    final_results = []

    for i, test in enumerate(test_cases):
        # Extract prompt regardless of input JSON structure
        if isinstance(test, dict) and "prompt" in test:
            prompt_test = test["prompt"]
        elif isinstance(test, str):
            prompt_test = test
        else:
            prompt_test = str(test)

        print(f"\n[{i + 1}/{len(test_cases)}] Processing: '{prompt_test}'")

        t0 = time.time()
        # generator points to either JSONStateMachine
        # or UnconstrainedGenerator
        generated_call = generator.generate_call(prompt_test)
        dt = time.time() - t0
        print(f"Completed in {dt:.2f} sec")

        # Construct final compliant JSON object
        # Strictly uses: "prompt", "name", "parameters"
        final_results.append({
            "prompt": prompt_test,
            "name": generated_call.get("name"),
            "parameters": generated_call.get("parameters", {})
        })

    # Exporting the final results ==============================
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print("\n=== Exporting Results ===")
    print(f"[SUCCESS]: All results are safely written to: \n{output_path}.")

    t_test_completed = time.time() - t_test_start
    print(f"\nTest prompts processed in {t_test_completed:.2f} sec. or "
          f"{(t_test_completed / 60):.2f} min.")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[FATAL ERROR]: {err}")
        sys.exit(1)
