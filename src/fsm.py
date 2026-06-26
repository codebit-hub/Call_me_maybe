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
        raw_output: Any = self.llm.encode(text)

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
        possible_functions: list[str] = list(self.registry.keys())

        name_to_tokens_map: dict[str, list[int]] = {}
        for func_name in possible_functions:
            name_to_tokens_map[func_name] = self._normalize_to_list(func_name)

        menu_text = ""
        for name, spec in self.registry.items():
            menu_text += f"- {name}: {spec.description}\n"

        prompt_text = (
            "Analyze the request and select the matching function name.\n\n"
            "AVAILABLE FUNCTIONS:\n"
            f"{menu_text}\n"
            f"Request: {prompt}\n"
            "``" + '`json\n{"function": "'
        )
        prompt_tokens = self._normalize_to_list(prompt_text)

        generated_tokens: list[int] = []
        current_token_index = 0

        while len(possible_functions) > 1 and current_token_index < 10:
            legal_next_tokens: set[int] = set()
            for func_name in possible_functions:
                tokens = name_to_tokens_map[func_name]
                if current_token_index < len(tokens):
                    legal_next_tokens.add(tokens[current_token_index])

            current_context = prompt_tokens + generated_tokens
            raw_logits = self.llm.get_logits_from_input_ids(current_context)
            logits = np.array(raw_logits)

            mask = np.zeros(self.masks.vocab_size, dtype=bool)
            for valid_id in legal_next_tokens:
                mask[valid_id] = True

            logits[~mask] = -np.inf

            winning_token = int(np.argmax(logits))
            generated_tokens.append(winning_token)
            current_token_index += 1

            surviving_functions = []
            for func_name in possible_functions:
                tokens = name_to_tokens_map[func_name]
                if current_token_index <= len(tokens):
                    if tokens[current_token_index - 1] == winning_token:
                        surviving_functions.append(func_name)

            possible_functions = surviving_functions

        if not possible_functions:
            possible_functions = list(self.registry.keys())
        return self.registry[possible_functions[0]]

    def _cast_number(self, val_str: str) -> float | int:
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
        args_so_far_str = json.dumps(resolved_args)[:-1]
        if resolved_args:
            args_so_far_str += ", "

        expected_type = func_spec.parameters[arg_name].type

        if expected_type == "string":
            prefix = f'"{arg_name}": "'
        else:
            prefix = f'"{arg_name}": '

        # Highly aggressive Few-Shot rules to override the model's biases
        if expected_type in ("number", "integer"):
            max_tokens = 30
            instruction = (
                f"Find the exact numerical value for '{arg_name}'.\n"
                "CRITICAL RULES:\n"
                "1. Do NOT calculate. Extract the literal digits.\n"
                "2. Include minus signs perfectly "
                "(e.g., if -500.00, output -500.00).\n"
                "3. Include exact decimals perfectly "
                "(e.g., if 0.0000000001, output 0.0000000001)."
            )
        elif expected_type == "boolean":
            max_tokens = 10
            instruction = (
                f"Find the boolean value for '{arg_name}' "
                "(true or false)."
            )
        else:
            max_tokens = 120
            instruction = (
                f"Extract the exact text for '{arg_name}'.\n"
                "CRITICAL RULES:\n"
                "1. If text contains internal quotes (\"), "
                "you MUST escape them as \\\".\n"
                "2. Do NOT invent regex. Extract the simple, "
                "literal target word (e.g., cat).\n"
                "3. Do NOT change capitalization. Copy exact casing.\n"
                "4. Preserve all {curly braces}, SQL syntax, "
                "and formatting perfectly."
            )

        mold_text = (
            f"User Request: {prompt}\n"
            "-----------------\n"
            f"Task: {instruction}\n"
            "-----------------\n"
            "Output JSON:\n"
            "``" + f'`json\n{{"function": "{func_spec.name}", '
            f'"arguments": {args_so_far_str}{prefix}'
        )

        context_tokens = self._normalize_to_list(mold_text)
        generated_value_tokens: list[int] = []

        for _ in range(max_tokens):
            current_context = context_tokens + generated_value_tokens
            raw_logits = self.llm.get_logits_from_input_ids(current_context)
            logits = np.array(raw_logits)

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

            winning_token = int(np.argmax(logits))
            generated_value_tokens.append(winning_token)

            if expected_type == "string":
                current_text = self.llm.decode(generated_value_tokens)

                clean_text = current_text.replace(
                    '\\\\', '__'
                ).replace('\\"', '__')

                if '"' in clean_text:
                    break
            else:
                is_comma = self.masks.comma[winning_token]
                is_brace = self.masks.close_brace[winning_token]
                if winning_token != 0 and (is_comma or is_brace):
                    break

        raw_string_value = self.llm.decode(generated_value_tokens).strip()

        if expected_type in ("number", "integer"):
            clean_num = raw_string_value.rstrip(",}")
            return self._cast_number(clean_num)

        elif expected_type == "boolean":
            clean_bool = raw_string_value.rstrip(",}").strip().lower()
            return clean_bool == "true"

        elif expected_type == "string":
            extracted = ""
            i = 0
            while i < len(raw_string_value):
                if (
                    raw_string_value[i] == '\\'
                    and i + 1 < len(raw_string_value)
                ):
                    extracted += raw_string_value[i:i+2]
                    i += 2
                elif raw_string_value[i] == '"':
                    break
                else:
                    extracted += raw_string_value[i]
                    i += 1

            try:
                return json.loads(f'"{extracted}"')
            except Exception:
                return extracted.replace('\\"', '"').replace('\\\\', '\\')

        return raw_string_value

    def generate_call(self, prompt: str) -> dict[str, Any]:
        spec = self._select_function(prompt)
        args_accumulator: dict[str, Any] = {}

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
    test_prompt = (
        "Calculate compound interest on a negative principal "
        "of -500.00 at 0.05 rate for 10 years"
    )

    print("Booting Finite State Machine (FSM) Environment...")
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
        print(json.dumps(final_call, indent=2))

    except Exception as err:
        print(f"\n[FATAL ERROR]: {err}")
        sys.exit(1)
