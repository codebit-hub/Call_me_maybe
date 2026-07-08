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
        candidates: list[str] = list(self.registry.keys())
        cand_tokens: dict[str, list[int]] = {}
        for func_name in candidates:
            cand_tokens[func_name] = self._normalize_to_list(func_name)

        menu_text = ""
        for name, spec in self.registry.items():
            menu_text += f"- {name}: {spec.description}\n"

        # Splitting the backticks ensures the UI markdown parser doesn't break
        context = (
            f"Available Functions:\n"
            f"{menu_text}\n"
            f"Request: {prompt}\n"
            "``" + "`json\n{\"name\": \""
        )

        base_ids: list[int] = self._normalize_to_list(context)
        current_seq: list[int] = []
        depth = 0

        # Increased depth cap to 20 to accommodate very long function names
        while len(candidates) > 1 and depth < 20:
            # Improved Pythonic set comprehension for allowed tokens
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

            mask = np.zeros(self.masks.vocab_size, dtype=bool)

            for tid in allowed_toks:
                mask[tid] = True

            logits[~mask] = -np.inf
            winning_id = int(np.argmax(logits))

            current_seq.append(winning_id)
            depth += 1

            candidates = [
                name
                for name in candidates
                if depth <= len(cand_tokens[name])
                and cand_tokens[name][depth - 1] == winning_id
            ]

        if candidates:
            chosen_name = candidates[0]
        else:
            chosen_name = list(self.registry.keys())[0]

        return self.registry[chosen_name]

    def _cast_number(self, val_str: str, expected_type: str) -> float | int:
        """Safely coerces a generated numeric string into a Python scalar."""
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
        """Generates an argument value constrained by its Pydantic schema."""
        # 1. Build the exact JSON scaffolding leading up to this parameter
        args_json_partial = json.dumps(resolved_args)[:-1]
        if resolved_args:
            args_json_partial += ", "

        expected_type = func_spec.parameters[arg_name].type

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

        context_ids: list[int] = self._normalize_to_list(context)
        val_toks: list[int] = []

        for _ in range(30):  # Token generation safety cap per argument
            raw_logits = self.llm.get_logits_from_input_ids(
                context_ids + val_toks
            )
            # Ensure it is explicitly a NumPy array
            # to prevent PyTorch indexing errors
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
                raw_text = self.llm.decode(val_toks + [winning_id])
                clean_text = raw_text.replace(
                    '\\\\', '__'
                ).replace('\\"', '__')
                if '"' in clean_text:
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

        raw_val_str = self.llm.decode(val_toks).strip()

        if expected_type in ("number", "integer"):
            clean_num = raw_val_str.rstrip(",}")
            return self._cast_number(clean_num, expected_type)

        elif expected_type == "boolean":
            clean_bool = raw_val_str.rstrip(",}").strip().lower()
            return clean_bool == "true"

        elif expected_type == "string":
            extracted = ""
            i = 0
            while i < len(raw_val_str):
                if raw_val_str[i] == '\\' and i + 1 < len(raw_val_str):
                    extracted += raw_val_str[i:i+2]
                    i += 2
                elif raw_val_str[i] == '"':
                    break
                else:
                    extracted += raw_val_str[i]
                    i += 1
            try:
                return json.loads(f'"{extracted}"')
            except Exception:
                return extracted.replace('\\"', '"').replace('\\\\', '\\')

        return raw_val_str

    def generate_call(self, prompt: str) -> dict[str, Any]:
        """Executes the complete FSM pipeline for a user prompt."""
        # Station 1: Resolve Target Function
        spec = self._select_function(prompt)

        # Station 2 & 3: Mold and Inject Arguments
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
        print(json.dumps(final_call, indent=2))

    except Exception as err:
        print(f"\n[FATAL ERROR]: {err}")
        sys.exit(1)
