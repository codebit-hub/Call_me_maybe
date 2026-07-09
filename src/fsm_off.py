import json
from typing import Any, cast
import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.schema import FunctionSpec


class UnconstrainedGenerator:
    """
    A control group generator that lets LLM process prompts
    without any FSM constraints. It relies only on
    prompt engineering and built-in LLM capabilities.
    """

    def __init__(
            self,
            llm: Small_LLM_Model,
            registry: dict[str, FunctionSpec]
    ) -> None:
        self.llm = llm
        self.registry = registry

    def _normalize_to_list(self, text: str) -> list[int]:
        """
        Forces LLM SDK output into a simple Python list.
        """
        # var text is system_prompt

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

        # If LLM returns multi-dimentional lists of ints
        if isinstance(data[0], list):
            return [int(x) for x in data[0]]
        return [int(cast(int, x)) for x in data]

    def generate_call(self, prompt: str) -> dict[str, Any]:
        """
        Runs greedy decoding without any vocabulary masks.
        """

        # Build a textual menu of functions and params for the prompt

        menu_text = ""
        for name, spec in self.registry.items():
            menu_text += f"- {name}: {spec.description}\n"
            for param_name, param_spec in spec.parameters.items():
                menu_text += f"    - {param_name} ({param_spec.type})\n"

        # Ask the LLM to write its output in a JSON file on its own.
        # Added strict instructions for EXACT keys required by subject.

        system_prompt = (
            "You are a function-calling assistant.\n"
            "Choose exactly one function from the catalog.\n"
            "Return only a JSON object with name and parameters.\n"
            "Do not include extra text."
            "AVAILABLE FUNCTIONS:\n"
            f"{menu_text}\n"
            f"User Request: {prompt}\n"
            "Output JSON:\n"
            "{"
        )

        # Encodes system prompt into validated tokens and returns
        # numeric tokens [tensors, ndarrays] normalized into list

        context_tokens = self._normalize_to_list(system_prompt)
        generated_value_tokens: list[int] = []

        # Run unconstrained generation up to a max of 150 tokens

        for _ in range(150):
            # Send current_context to LLM for processing and decoding

            current_context = context_tokens + generated_value_tokens
            raw_logits = self.llm.get_logits_from_input_ids(current_context)

            # Pure greedy decoding: pick the highest probability token only

            winning_token = int(np.argmax(raw_logits))
            generated_value_tokens.append(winning_token)

            # Check if LLM outputted a closing brace making valid JSON
            # Early Stopping Circuit Breaker
            # Instead of waiting for all 150 tokens every single time,
            # we check the output after every single token is generated
            # and abort as soon as '}' reached.

            current_text = "{" + self.llm.decode(generated_value_tokens)
            try:
                # if } present, create JSON output file and break
                # json.loads() converts JSON input into python object
                # endswith() performs a check for ending "}"
                if current_text.strip().endswith("}"):
                    json.loads(current_text)
                    break

            except json.JSONDecodeError:
                # Keep generatig if it is not valid JSON yet
                pass

        # Once for loop finishes (reached '}', broke, maxed out 150 tokens),
        # we try parsing LLM's messy output

        raw_output = "{" + self.llm.decode(generated_value_tokens)

        try:
            # Clean up potential markdown formatting
            clean_output = raw_output.replace("```json", "")
            clean_output = clean_output.replace("```", "")
            clean_output = clean_output.strip()

            # json.loads() returns Any type. To meet mypy:
            return cast(dict[str, Any], json.loads(clean_output))

        except json.JSONDecodeError:
            # if it failed to generate valid JSON, return raw hallucination
            # Ensure it STILL conforms to the 'name' and 'parameters' structure
            return {
                "name": "error_parsing_failed",
                "parameters": {
                    "error": "LLM failed to produce a valid JSON output file.",
                    "raw_output": raw_output
                }
            }
