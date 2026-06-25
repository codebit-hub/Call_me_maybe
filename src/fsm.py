import json
from typing import Any
import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.schema import FunctionSpec, load_schema
from src.vocab import VocabMasks


class JSONStateMachine:
    """
    Executes deterministic JSON scaffolding with logit injection.
    """

    def __init__(
        self,
        llm: Small_LLM_Model,
        registry: dict[str, FunctionSpec],
        masks: VocabMasks,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.masks = masks

    # This function receives a function name for conversion
    # into token IDs - two digits [[101, 452]]

    def _normalize_to_list(self, text: str) -> list[int]:
        """
        Sometimes the LLM SDK returns a PyTorch Tensor, sometimes a List.
        This bridge catches the output and forces it into a simple Python list.
        """

        raw_output: Any = self.llm.encode(text)

        # There is requsted conversion of a function name into tokenid.
        # LLM can return a pytorch tensor, 1D list, 2D list, etc.
        # No matter what we get, we check is there attribute tolist(),
        # if yes, run it to make list.

        if hasattr(raw_output, "tolist"):
            data = raw_output.tolist()
        elif isinstance(raw_output, list):
            data = raw_output
        else:
            return []

        if not isinstance(data, list) or not data:
            return []

        # If the SDK returned a 2D batch tensor e.g. [[101, 452]],
        # pluck row 0
        if isinstance(data[0], list):
            return [int(x) for x in data[0]]
        # No matter what LLM spits out, we normalize it to a list of ints
        # e.g. [120, 482, 999]
        return [int(x) for x in data]

    def _select_function(self, prompt: str) -> FunctionSpec:
        """
        STATION 1: FUNCTION NAME SELECTION
        Forces the LLM to choose a valid function name token-by-token.
        """

        # Get a list of function names from the registry

        possible_functions: list[str] = list(self.registry.keys())

        # Step 1: Pre-calculate the token IDs for every valid function name.
        # Example: "fn_add" might become tokens [101, 2045]
        # LLM understands only digits and math. Everything gets tokenized,
        # i.e. all input strings encoded into ints

        name_to_tokens_map: dict[str, list[int]] = {}
        for func_name in possible_functions:
            name_to_tokens_map[func_name] = self._normalize_to_list(func_name)

        # Step 2: Build the prompt forcing the LLM to start writing a JSON.
        # instead of a human-like text.

        prompt_text = (
            "Analyze the request and select the matching function name.\n"
            f"Request: {prompt}\n"
            "``" + '`json\n{"function": "'
        )
        prompt_tokens = self._normalize_to_list(prompt_text)

        # Step 3: Loop and generate tokens until 1 valid function remains
        # current_token_index < 10 is hardcoded guard - no function is
        # longer than 10 tokens - prevents hallucinations

        generated_tokens: list[int] = []
        current_token_index = 0

        while len(possible_functions) > 1 and current_token_index < 10:
            # Find all legal next tokens based on remaining valid functions
            legal_next_tokens: set[int] = set()
            for func_name in possible_functions:
                tokens = name_to_tokens_map[func_name]
                if current_token_index < len(tokens):
                    next_valid_token = tokens[current_token_index]
                    legal_next_tokens.add(next_valid_token)

            # Ask the LLM to score the next token based on current context
            current_context = prompt_tokens + generated_tokens
            raw_logits = self.llm.get_logits_from_input_ids(current_context)
            logits = np.array(raw_logits)

            # Apply the mask: Set all ILLEGAL tokens to negative infinity
            mask = np.zeros(self.masks.vocab_size, dtype=bool)
            for valid_id in legal_next_tokens:
                mask[valid_id] = True

            logits[~mask] = -np.inf

            # Pick the highest scoring allowed token
            winning_token = int(np.argmax(logits))
            generated_tokens.append(winning_token)
            current_token_index += 1

            # Step 4: Drop any functions that don't match the token we picked
            surviving_functions = []
            for func_name in possible_functions:
                tokens = name_to_tokens_map[func_name]
                # Check if this function is long enough AND matches the token
                if current_token_index <= len(tokens):
                    last_picked = tokens[current_token_index - 1]
                    if last_picked == winning_token:
                        surviving_functions.append(func_name)

            possible_functions = surviving_functions

        # Step 5: Return the one function that survived the elimination process
        if not possible_functions:
            possible_functions = list(self.registry.keys())
        chosen_name = possible_functions[0]
        return self.registry[chosen_name]

    def _cast_number(self, val_str: str) -> float | int:
        """Safely coerces a generated numeric string into a Python scalar."""
        clean = val_str.strip()
        try:
            if "." in clean or "e" in clean.lower():
                return float(clean)
            return int(clean)
        except ValueError:
            return 0

    def _generate_arg_value(
        self,
        prompt: str,
        func_spec: FunctionSpec,
        arg_name: str,
        resolved_args: dict[str, Any],
    ) -> Any:
        """
        STATION 2 & 3: THE MOLD AND INJECTOR
        Builds the JSON scaffold for one argument and lets the LLM fill it.
        """
        # Step 1: Turn the arguments we already solved into a string.
        # Example output: {"a": 265
        args_so_far_str = json.dumps(resolved_args)[:-1]
        if resolved_args:
            args_so_far_str += ", "

        expected_type = func_spec.parameters[arg_name].type

        # Step 2: Prepare the prefix for the specific type
        if expected_type == "string":
            prefix = f'"{arg_name}": "'  # Includes opening quote
        else:
            prefix = f'"{arg_name}": '   # No quote for numbers or bools

        # Step 3: Combine everything into the "Concrete Mold"
        mold_text = (
            f"User request: {prompt}\n"
            "Target JSON:\n"
            "``" + f'`json\n{{"function": "{func_spec.name}", '
            f'"arguments": {args_so_far_str}{prefix}'
        )
        context_tokens = self._normalize_to_list(mold_text)
        generated_value_tokens: list[int] = []

        # Step 4: Let the LLM generate the value token by token (max 25)
        for _ in range(25):
            current_context = context_tokens + generated_value_tokens
            raw_logits = self.llm.get_logits_from_input_ids(current_context)
            logits = np.array(raw_logits)

            # Step 5: Apply strict grammar masks based on parameter type
            if expected_type == "number":
                # Allow only digits, commas (end arg), or braces (end JSON)
                mask = (
                    self.masks.digits
                    | self.masks.comma
                    | self.masks.close_brace
                )
                logits[~mask] = -np.inf

            elif expected_type == "boolean":
                # Allow only true/false, commas, or braces
                mask = (
                    self.masks.booleans
                    | self.masks.comma
                    | self.masks.close_brace
                )
                logits[~mask] = -np.inf

            winning_token = int(np.argmax(logits))
            token_string = self.llm.decode([winning_token]).strip()

            # Step 6: Check if the LLM naturally completed the argument
            if expected_type == "string":
                # A string is finished when the LLM outputs a closing quote
                if '"' in token_string:
                    break
            else:
                # Numbers/Bools finish when the LLM spits a comma or brace
                is_comma = self.masks.comma[winning_token]
                is_brace = self.masks.close_brace[winning_token]
                if winning_token != 0 and (is_comma or is_brace):
                    break

            generated_value_tokens.append(winning_token)

        # Step 7: Decode the collected tokens into a final python value
        raw_string_value = self.llm.decode(generated_value_tokens).strip()

        if expected_type == "number":
            return self._cast_number(raw_string_value)
        elif expected_type == "boolean":
            return raw_string_value.lower() == "true"
        return raw_string_value

    def generate_call(self, prompt: str) -> dict[str, Any]:
        """Executes the complete FSM pipeline for a user prompt."""

        # 1. Ask the Router to find the function
        spec = self._select_function(prompt)
        args_accumulator: dict[str, Any] = {}

        # 2. Iterate through expected parameters and inject values
        for arg_name in spec.parameters.keys():
            val = self._generate_arg_value(
                prompt,
                spec,
                arg_name,
                args_accumulator,
            )
            args_accumulator[arg_name] = val

        return {
            "function": spec.name,
            "arguments": args_accumulator,
        }


if __name__ == "__main__":
    import sys
    import time

    schema_path = "data/input/functions_definition.json"
    test_prompt = "What is the sum of 265 and 345?"

    print("Booting Finite State Machine (FSM) Environment...")
    try:
        # Parses JSON file into [str, FunctionSpec] registry
        registry = load_schema(schema_path)

        # Inializes the provided LLM
        model = Small_LLM_Model(device="cpu")

        # Retreives entire LLM's vocabulary token by token
        # then validates and gategorizes tokens by type
        # into respective ndarrays for fast bitwise masking
        masks = VocabMasks(model)

        # Initiates MSM, processes the prompt
        fsm = JSONStateMachine(model, registry, masks)

        print(f"\n[RUNNING PROMPT]: '{test_prompt}'")
        t0 = time.time()
        final_call = fsm.generate_call(test_prompt)
        dt = time.time() - t0

        print(f"[SUCCESS]: Call assembled in {dt:.2f} seconds.\n")
        # Converts dict into JSON compliant object with 2 tabs indent
        print(json.dumps(final_call, indent=2))

    except Exception as err:
        print(f"\n[FATAL ERROR]: {err}")
        sys.exit(1)
