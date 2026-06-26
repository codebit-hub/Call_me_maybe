import json
import os
import time
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.schema import load_schema
from src.vocab import VocabMasks
from src.fsm import JSONStateMachine


def main() -> None:
    """
    The main program
    """

    # Accessing files ==================================

    schema_path = "data/input/functions_definition.json"
    test_path = "data/input/function_calling_tests.json"

    # Validates/Creates the output path
    # makedirs creates the entire path, recurrsively
    # if the target directory exists, exist_ok=True just
    # quitely accepts the path.

    output_dir = "data/output"
    output_path = f"{output_dir}/function_calling_result.json"
    os.makedirs(output_dir, exist_ok=True)

    # Boot Up the Environment ==========================

    print("=== Booting Environment ===")

    # Load the function definitions from JSON file to registry

    registry = load_schema(schema_path)
    print(f"+ Loaded {len(registry)} functions from schema.")

    # Boot the LLM on the CPU

    model = Small_LLM_Model(device="cpu")
    print("+ LLM has been deployed.")

    # Calculate the vocabulary masks

    masks = VocabMasks(model)
    print("+ Vocabulary masks have been aggrigated and set.")

    # Boot up Finite State Machine (FSM) - Communicator with LLM

    fsm = JSONStateMachine(model, registry, masks)
    print("+ FSM has been deployed and is ready.")

    # Load the test cases ======================================

    print("\n=== Processing Prompts ===")

    # Loading test prompts

    print("+ Loading test prompts.")
    with open(test_path, "r", encoding="utf-8") as f:
        # json.load() reads the file into a python list
        test_cases = json.load(f)
    print(f"+ Loaded {len(test_cases)} tests to process.")

    # Starting timer for test processing
    t_test_start = time.time()

    # Procesing the loaded tests one by one
    # If JSON file provides a dict, we grab "prompt"'s value.
    # if it is a raw string, we use it directly.

    final_results = []
    for i, test in enumerate(test_cases):
        if isinstance(test, dict) and "prompt" in test:
            prompt_test = test["prompt"]
            test_id = test.get("id", f"test_{i}")
        elif isinstance(test, str):
            prompt_test = test
            test_id = f"test_{i}"
        else:
            prompt_test = str(test)
            test_id = f"test_{i}"
        print(f"\n[{i + 1}/{len(test_cases)}] Processing: '{prompt_test}'")

        # Starting the timer
        t0 = time.time()
        generated_call = fsm.generate_call(prompt_test)
        dt = time.time() - t0
        print(f"Completed in {dt:.2f} sec")

        # Save the results in the correct output format

        final_results.append({
            "id": test_id,
            "prompt": prompt_test,
            "result": generated_call
        })

    # Exporting the final results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\n=== Exporting Results ===")
    print(
        "[SUCCESS]: All results are safely written to: \n"
        f"{output_path}.")
    print(f"Review the output JSON file in {output_dir}.")

    # Stopping timer for test processing
    t_test_completed = time.time() - t_test_start
    print(f"Test prompts processed in {t_test_completed:.2f} sec.")


if __name__ == "__main__":
    import sys

    try:
        main()

    except Exception as err:
        print(f"\n[FATAL ERROR]: {err}")
        sys.exit(1)
