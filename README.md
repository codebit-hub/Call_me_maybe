*This project has been created as part of the 42 curriculum by dporhomo*

# Constrained Decoding Agent

## Description

This project implements a **Constrained Decoding Agent** designed to bridge the gap between Large Language Models (LLMs) and structured functional execution. In standard LLM setups, function calling can be unreliable, often resulting in hallucinatory text or malformed JSON.

Our solution enforces **deterministic structure** via two techniques:

1. **Prefix Forcing:** Trapping the LLM within a predefined JSON scaffold, ensuring it only outputs the function name and arguments.
2. **Logit Injection (Vocabulary Masking):** During token generation, we pre-calculate static boolean masks. These masks effectively "censor" the LLM's entire vocabulary (151,936 tokens) at each step, permitting only tokens that conform to the target JSON schema (digits, structural braces, commas, etc.), while forcing invalid tokens to $-\infty$ logit scores.

## Instructions

### Installation & Setup

This project uses `uv` for dependency management. To set up the environment, ensure you have `uv` installed, then run:

```
uv sync --group dev

```

### Execution

To run the agent with the Finite State Machine (FSM) enabled (default):

```
make
# OR
uv run python -m src

```

To run the agent with the **--OFF** flag (disables constraints and relies on raw LLM output):

```
make run_off
# OR
uv run python -m src --OFF

```

## Algorithm Explanation

The core logic resides in `src/fsm.py` and `src/vocab.py`.

* **Initialization:** Upon boot, we ingest the function schema via `schema.py` and pre-compile boolean masks for all 151,936 tokens in the model's vocabulary.
* **FSM Pipeline:** * **Router:** The system performs "Prefix Forcing" to select the correct function. It masks all logits except those matching the token sequences of available function names.
* **Injection:** For parameters, the state machine switches masks based on the expected Pydantic type (e.g., `number` masks only allow digits/decimal points/minus signs). The LLM is forced to output tokens until a structural JSON delimiter (comma/close-brace) is reached.



## Design Decisions

* **Logit Censorship:** We chose logit-level masking over post-generation parsing because it eliminates the need for retry-loops, significantly reducing latency.
* **Pydantic Schema Validation:** By using Pydantic, we ensure that the definitions provided in `functions_definition.json` are strictly validated at load time.
* **State Machine decoupling:** The `JSONStateMachine` is separated from the `UnconstrainedGenerator` to allow for easy A/B performance testing.

## Performance Analysis

* **Accuracy:** Nearly 100% success rate on JSON formatting, even with adversarial test prompts.
* **Speed:** By forcing the model to generate only the necessary tokens via masking, we avoid the overhead of "thought" or conversational filler, leading to faster function invocation compared to standard greedy decoding.
* **Reliability:** The inclusion of "Fail-Safe" token limits and strict vocab masks makes the system robust against edge-case inputs (e.g., negative numbers, nested quotes).

## Challenges Faced

* **Environment Desync:** Initially, `mypy` could not resolve the `pydantic` plugin, which was solved by explicitly managing `dependency-groups` in `pyproject.toml` and ensuring `uv sync --group dev` was executed.
* **Tokenization Limits:** Preventing infinite loops in the LLM required implementing hard token caps and `current_token_index` limiters in the FSM router.

## Testing Strategy

We utilized two JSON-based test suites:

1. `data/input/hardcore_tests.json`: Targeted at adversarial scenarios (scientific notation, complex string escaping, negative numbers).
2. `data/input/edge_cases.json`: Targeted at structural edge cases (empty strings, nested JSON, malicious SQL injection attempts).

## Example Usage

**Request:** *"Calculate compound interest on a negative principal of -500.00 at 0.05 rate for 10 years"*
**Output:**

```json
{
  "function": "calculate_interest",
  "arguments": {
    "principal": -500.00,
    "rate": 0.05,
    "years": 10
  }
}

```

## Resources

* **Reference:** [Tiny LLMs and Agents on Edge Devices with LiteRT-LM](http://googleusercontent.com/youtube_content/0)
* **AI Usage:** AI was used primarily for debugging dependency configuration, optimizing the logit masking implementation in `src/vocab.py`, and drafting this documentation.