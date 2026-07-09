import json
from typing import Any
import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.schema import FunctionSpec, load_schema
from src.vocab import VocabMasks


class JSONStateMachine:
    """Executes deterministic JSON scaffolding with logit injection."""

    def __init__(
        self,
        llm: Small_LLM_Model,
        registry: dict[str, FunctionSpec],
        masks: VocabMasks,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.masks = masks

    def _normalize_to_list(self, text: str) -> list[int]:
        """
        Forces LLM SDK output into a simple Python list.
        """
        raw_output: Any = self.llm.encode(text)

        # We dont know what raw_output is: tensor, ndarray, list
        # we check for method name 'tolist()' need for conversion
        # if the tolist() method present, we use it for conversion

        if hasattr(raw_output, "tolist"):
            data = raw_output.tolist()
        elif isinstance(raw_output, list):
            data = raw_output
        else:
            return []

        if not isinstance(data, list) or not data:
            return []

        if isinstance(data[0], list):
            return [int(x) for x in data[0]]
        return [int(x) for x in data]

    def _select_function(self, prompt: str) -> FunctionSpec:
        """
        Selects the correct function name
        """
        # valid candidates come from the registry.keys()
        candidates: list[str] = list(self.registry.keys())

        # encode valid function candidates for LLM into
        # normalized 1D list
        cand_tokens: dict[str, list[int]] = {}
        for func_name in candidates:
            cand_tokens[func_name] = self._normalize_to_list(func_name)

        # building a string menu_text based on function name and specs
        # from the registry
        menu_text = ""
        for name, spec in self.registry.items():
            menu_text += f"- {name}: {spec.description}\n"

        # Splitting the backticks ensures the UI markdown
        # parser doesn't break
        context = (
            "You are a function-calling assistant.\n"
            "Choose exactly one function from the catalog.\n"
            "Return only a JSON object with name and parameters.\n"
            "Do not include extra text."
            f"Available Functions:\n"
            f"{menu_text}\n"
            f"Request: {prompt}\n"
            "``" + "`json\n{\"name\": \""
        )

        # encode context into numeral ids
        base_ids: list[int] = self._normalize_to_list(context)
        current_seq: list[int] = []
        depth = 0

        # Increased depth cap to 20 to accommodate very long function names
        # From all func candidates, we choose only correct functions
        # depth represents the current token index, which less than 20,
        # prevents hallucinations.

        while len(candidates) > 1 and depth < 20:
            # Improved Pythonic set comprehension for allowed tokens
            # Set is used to keep only unique tokens.
            # For only allowed function names in candidates with
            # their depth less than number of tokens assigned to that name
            # in cand_tokens[name]
            allowed_toks: set[int] = {
                cand_tokens[name][depth]
                for name in candidates
                if depth < len(cand_tokens[name])
            }

            raw_logits = self.llm.get_logits_from_input_ids(
                base_ids + current_seq
            )
            # Ensure it is explicitly a NumPy array
            # to prevent PyTorch indexing errors
            logits = np.array(raw_logits)

            # Create a mask for the size of LLM tokens, set it to 0
            mask = np.zeros(self.masks.vocab_size, dtype=bool)

            # For all allowed num tokens in allowed tokenized allowed_toks
            # set the mask[token id] to True and set the rest to -inf
            for tid in allowed_toks:
                mask[tid] = True

            # logits at reversed mask are set to -inf
            # the best logit is chosen from only valid values
            logits[~mask] = -np.inf
            winning_id = int(np.argmax(logits))

            # add the winning id to current_seq and increase depth by 1
            # ready for next token
            current_seq.append(winning_id)
            depth += 1

            # Select the best candidate 'function name' as str
            # if depth determines valid length of function name
            # 0-indexed cand_tokens
            # if at current depth is winning_id, that function name
            # gets chosen
            candidates = [
                name
                for name in candidates
                if depth <= len(cand_tokens[name])
                and cand_tokens[name][depth - 1] == winning_id
            ]

        if candidates:
            chosen_name = candidates[0]
        else:
            # fail-save: if there isn't any match,
            # it chooses the first function name in registry
            chosen_name = list(self.registry.keys())[0]

        return self.registry[chosen_name]

    def _cast_number(self, val_str: str, expected_type: str) -> float | int:
        """
        Safely coerces a generated numeric string into a Python scalar.
        """
        # If type = "number", return float, else integer
        clean = val_str.strip()
        try:
            if expected_type == "number":
                return float(clean)
            return int(float(clean))
        except ValueError:
            return 0.0 if expected_type == "number" else 0

    def _generate_arg_value(
        self,
        prompt: str,
        func_spec: FunctionSpec,
        arg_name: str,
        resolved_args: dict[str, Any],
    ) -> Any:
        """
        Generates constrained argument values for parameters.
        """
        # 1. Build the exact JSON scaffolding leading up to this parameter
        # If resolved_args contain value, {"a": 2},
        # the last char sliced [:-1] is '}', it json.dumps current
        # resolved_args and if they exist, adds ','
        args_json_partial = json.dumps(resolved_args)[:-1]
        if resolved_args:
            args_json_partial += ", "

        # check the parameter type from FunctionSpec
        expected_type = func_spec.parameters[arg_name].type

        # sets prefix that ends with '"' for str or without '"'if else
        if expected_type == "string":
            prefix = f'"{arg_name}": "'
        else:
            prefix = f'"{arg_name}": '

        # Retained optimized Zero-Shot prompt to prevent attention dilution
        context = (
            f"User request: {prompt}\n"
            f"Target JSON:\n"
            "``" + f"`json\n{{\"name\": \"{func_spec.name}\", "
            f"\"parameters\": {args_json_partial}{prefix}"
        )

        # encode tokens from the context
        context_ids: list[int] = self._normalize_to_list(context)
        val_toks: list[int] = []

        # LLM to generate the value for a specific parameter
        # one token at a time, applying strict rules at every single step.
        # Range cap was 30-50

        for _ in range(50):
            # Feed LLM tokenized promt and valid tokens so far
            raw_logits = self.llm.get_logits_from_input_ids(
                context_ids + val_toks
            )
            # Ensure it is explicitly a NumPy array
            # to prevent PyTorch indexing errors
            # It contains all 151936 possible logits from LLM
            logits = np.array(raw_logits)

            # Apply strict type masking
            # while explicitly allowing closure tokens
            if expected_type in ("number", "integer"):
                mask = (
                    self.masks.digits
                    | self.masks.comma
                    | self.masks.close_brace
                )
                logits[~mask] = -np.inf
            elif expected_type == "boolean":
                mask = (
                    self.masks.booleans
                    | self.masks.comma
                    | self.masks.close_brace
                )
                logits[~mask] = -np.inf

            winning_id = int(np.argmax(logits))

            # Check for natural grammar termination delimiters
            if expected_type == "string":
                # Handle internal escapes for strings
                # if it is " then break
                raw_text = self.llm.decode(val_toks + [winning_id])
                if '\n' in raw_text:
                    break

                clean_text = raw_text.replace(
                    '\\\\', '__'
                ).replace('\\"', '__')
                # ^^Instead of breaking on the FIRST quote, wait until the LLM
                # outputs the closing quote AND
                # the next JSON structural token (, or })
                stripped = clean_text.rstrip()
                if stripped.endswith(',') or stripped.endswith('}'):
                    # Check if the character right before the comma/brace is
                    # the closing quote
                    if stripped[:-1].rstrip().endswith('"'):
                        val_toks.append(winning_id)
                        break
            else:
                if winning_id != 0 and (
                    self.masks.comma[winning_id]
                    or self.masks.close_brace[winning_id]
                ):
                    val_toks.append(winning_id)
                    break

            val_toks.append(winning_id)

        # decode the validated tokens from LLM
        raw_val_str = self.llm.decode(val_toks).strip()

        if expected_type in ("number", "integer"):
            clean_num = raw_val_str.rstrip(",}")
            return self._cast_number(clean_num, expected_type)

        elif expected_type == "boolean":
            clean_bool = raw_val_str.rstrip(",}").strip().lower()
            return clean_bool == "true"

        elif expected_type == "string":
            # Safely extract the inner string from the LLM's raw output
            clean_val = raw_val_str.rstrip()

            # 1. Remove the trailing comma or brace if the LLM generated one
            if clean_val.endswith(',') or clean_val.endswith('}'):
                clean_val = clean_val[:-1].rstrip()

            # 2. Remove the final closing quote
            if clean_val.endswith('"'):
                clean_val = clean_val[:-1]

            # 3. Return the exact inner text!
            # If the LLM successfully escaped everything, json.loads works.
            # If it forgot to escape internal quotes, json.loads will fail,
            # we catch the error and just return the raw string as generated
            try:
                return json.loads(f'"{clean_val}"')
            except Exception:
                return clean_val.replace('\\"', '"').replace('\\\\', '\\')

        return raw_val_str

    def generate_call(self, prompt: str) -> dict[str, Any]:
        """Executes the complete FSM pipeline for a user prompt."""

        # Resolve Target Function
        spec = self._select_function(prompt)

        # Mold and Inject Arguments
        args_accumulator: dict[str, Any] = {}
        for arg_name in spec.parameters.keys():
            val = self._generate_arg_value(
                prompt, spec, arg_name, args_accumulator
            )
            args_accumulator[arg_name] = val
        return {
            "name": spec.name,
            "parameters": args_accumulator,
        }


if __name__ == "__main__":
    import sys
    import time

    schema_path = "data/input/functions_definition.json"
    test_prompt = "What is the sum of 265 and 345?"

    print("Booting FSM Environment...")
    try:
        registry = load_schema(schema_path)
        model = Small_LLM_Model(device="cpu")
        masks = VocabMasks(model)

        fsm = JSONStateMachine(model, registry, masks)

        print(f"\n[RUNNING PROMPT]: '{test_prompt}'")
        t0 = time.time()
        final_call = fsm.generate_call(test_prompt)
        dt = time.time() - t0

        print(f"[SUCCESS]: Call assembled in {dt:.2f} seconds.\n")
        print(json.dumps(final_call, indent=4))

    except Exception as err:
        print(f"\n[FATAL ERROR]: {err}")
        sys.exit(1)
