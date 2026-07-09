*This project has been created as part of the 42 curriculum by dporhomo*

# Call Me Maybe
**Constrained Decoding Agent**

## Description

This project implements a **Constrained Decoding Agent** designed to bridge the gap between Large Language Models (LLMs) and structured functional execution. In standard LLM setups, function calling can be unreliable, often resulting in hallucinatory text or malformed JSON.

Our solution enforces **deterministic structure** via two techniques:

1. **Prefix Forcing:** Trapping the LLM within a predefined JSON scaffold, ensuring it only outputs the function name and arguments.
2. **Logit Injection (Vocabulary Masking):** During token generation, we pre-calculate static boolean masks. These masks effectively "censor" the LLM's entire vocabulary (151,936 tokens) at each step, permitting only tokens that conform to the target JSON schema (digits, structural braces, commas, etc.), while forcing invalid tokens to $-\infty$ logit scores.

## Instructions

### Installation & Setup

The operation of the agent requires at least 4 GB of storage for caching files: PyTorch (~2.4GB), the Hugging Face Hub cache (~1.5GB for Qwen3-0.6B), and the compiled wheel caches defaults to ~/.cache/ located on the root drive.
If your current root directory is large enouph you can proceed simply by executing 'make run'.
Otherwise, customize the directory for storing the cache files, as follows:

In the parent directory, create another directory called Cache for storing cache files, as follows:
  - cache/
    - huggingface/
    - uv/
    - pip/

Provide the following entries at the very bottom of the shell configuration files, as follows:
Open the configuration files:

```bash
vim ~/.zshrc
```
or
```bash
vim ~/.bashrc
```

At the bottom provide this text:
```
export HF_HOME="/path_to_your_project/Call_me_maybe/cache/huggingface"
export UV_CACHE_DIR="/path_to_your_project/Call_me_maybe/cache/uv"
export PIP_CACHE_DIR="path_to_your_project/Call_me_maybe/cache/pip"
```

After that run:

```bash
source ~/.bashrc
```
or
```bash
source ~/.zshrc
```

### Execution

To run the agent with the Finite State Machine (FSM) enabled (default):

```bash
make run
# OR
uv run python -m src

```

To run the agent with the **--off** flag (disables constraints and relies on raw LLM output):

```bash
make run_off
# OR
uv run python -m src --off

```

By default, the agent will process the predefined input file located at data/input/functions_calling_tests.json and output it into data/output/function_calling_result.json.
To run the agent with another files, for example, run the program with the following syntax as follows:
uv run python -m src [-functions_definition <function_definition_file>]
[-input <input_file>] [-output <output_file>]

```dash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests_easy.json --output data/output/my_function_calling_result.json
```

### Expected Output (regular run)
```bash
make run
Running Constrained Decoding Agent (FSM ON)...
uv run python -m src
      Built llm-sdk @ file:///the path to your project directory/llm_sdk
Installed 60 packages in 11.59s
=== Booting Environment ===
+ Loaded 11 functions from schema.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
config.json: 100%|████████████████████████████████████████████████████| 726/726 [00:00<00:00, 1.56MB/s]
tokenizer_config.json: 100%|██████████████████████████████████████| 9.73k/9.73k [00:00<00:00, 9.74MB/s]
vocab.json: 100%|█████████████████████████████████████████████████| 2.78M/2.78M [00:00<00:00, 20.0MB/s]
merges.txt: 100%|█████████████████████████████████████████████████| 1.67M/1.67M [00:00<00:00, 72.2MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████| 11.4M/11.4M [00:01<00:00, 8.92MB/s]
model.safetensors: 100%|██████████████████████████████████████████| 1.50G/1.50G [01:28<00:00, 16.9MB/s]
Loading weights: 100%|███████████████████████████████████████████████| 311/311 [00:13<00:00, 23.81it/s]
generation_config.json: 100%|██████████████████████████████████████████| 239/239 [00:00<00:00, 550kB/s]
+ LLM has been deployed.
+ Vocabulary masks have been aggrigated and set.
+ FSM has been deployed and is ready.

=== Processing Prompts ===
+ Loading test prompts from data/input/function_calling_tests.json.
+ Loaded 11 tests to process.

[1/11] Processing: 'What is the sum of 2 and 3?'
Completed in 3.80 sec

[2/11] Processing: 'What is the sum of 265 and 345?'
Completed in 5.95 sec

[3/11] Processing: 'Greet shrek'
Completed in 3.12 sec

[4/11] Processing: 'Greet john'
Completed in 2.73 sec

[5/11] Processing: 'Reverse the string 'hello''
Completed in 2.64 sec

[6/11] Processing: 'Reverse the string 'world''
Completed in 2.80 sec

[7/11] Processing: 'What is the square root of 16?'
Completed in 3.40 sec

[8/11] Processing: 'Calculate the square root of 144'
Completed in 3.95 sec

[9/11] Processing: 'Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS'
Completed in 15.15 sec

[10/11] Processing: 'Replace all vowels in 'Programming is fun' with asterisks'
Completed in 10.44 sec

[11/11] Processing: 'Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat''
Completed in 9.82 sec

=== Exporting Results ===
[SUCCESS]: All results are safely written to:
data/output/function_calling_result.json.

Test prompts processed in 63.83 sec. or 1.06 min.
```

### Partial Tests of Each Module
Each individual module is provided with a separate main() function that can test its operation.

Testing parsing of the function definition JSON file:

```bash
uv run python -m src.schema
```

Testing creation of fast masking NumPy ndarrays:

```bash
uv run python -m src.vocab
```

Testing operation of the Finite State Machine:

```bash
uv run python -m src.fsm
```

### Expected Output of Partial Module Tests
```bash
uv run python -m src.schema
Attempting to ingest schema from: data/input/functions_definition.json...
[SUCCESS]: Indexed 11 functions.

--fn_add_numbers (2 params)
    --a: <number>
    --b: <number>
--fn_greet (1 params)
    --name: <string>
--fn_reverse_string (1 params)
    --s: <string>
--fn_get_square_root (1 params)
    --a: <number>
--fn_substitute_string_with_regex (3 params)
    --source_string: <string>
    --regex: <string>
    --replacement: <string>
--fn_multiply_numbers (2 params)
    --a: <number>
    --b: <number>
--fn_is_even (1 params)
    --n: <integer>
--fn_calculate_compound_interest (3 params)
    --principal: <number>
    --rate: <number>
    --years: <integer>
--fn_execute_sql_query (2 params)
    --query: <string>
    --database: <string>
--fn_read_file (2 params)
    --path: <string>
    --encoding: <string>
--fn_format_template (1 params)
    --template: <string>
```

```bash
uv run python -m src.vocab
Initializing Small_LLM_Model on CPU...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████| 311/311 [00:00<00:00, 926.84it/s]
Model bound to HW in 2.37 seconds.
Compiling vocab truth tables...
[SUCCESS]: Processed 151,936 tokens in 0.40s.

 ├─ Valid Numeric Tokens: 222
 ├─ Valid Boolean Tokens: 6
 ├─ Null Tokens: 3
 ├─ Open Brace ('{') Tokens: 15
 ├─ Close Brace ('}') Tokens: 24
 ├─ Colon (':') Tokens: 13
 ├─ Comma (',') Tokens: 11
 └─ Quote ('"') Tokens: 12
```

## Algorithm Explanation

The core logic resides in `src/fsm.py` and `src/vocab.py`.

* **Initialization:** Upon boot, we ingest the function schema via `schema.py` and pre-compile boolean masks for all 151,936 tokens in the model's vocabulary.
* **FSM Pipeline:** * **Router:** The system performs "Prefix Forcing" to select the correct function. It masks all logits except those matching the token sequences of available function names.
* **Injection:** For parameters, the state machine switches masks based on the expected Pydantic type (e.g., `number` masks only allow digits/decimal points/minus signs). The LLM is forced to output tokens until a structural JSON delimiter (comma/close-brace) is reached.



## Design Decisions

* **Logit Censorship:** We chose logit-level masking over post-generation parsing because it eliminates the need for retry-loops, significantly reducing latency.
* **Pydantic Schema Validation:** By using Pydantic, we ensure that the definitions provided in `functions_definition.json` are strictly validated at load time.
* **Zero-Shot Prompting:** Small LLMs heavily suffer from "attention dilution" when overloaded with instructions. By relying on the structural constraints of the FSM masks, we successfully minimized the textual prompts, significantly increasing accuracy.
* **State Machine decoupling:** The `JSONStateMachine` is separated from the `UnconstrainedGenerator` to allow for easy A/B performance testing.

## Performance Analysis

* **Accuracy:** Nearly 100% success rate on JSON formatting, even with adversarial test prompts.
* **Speed:** By forcing the model to generate only the necessary tokens via masking, we avoid the overhead of "thought" or conversational filler, leading to faster function invocation compared to standard greedy decoding.
* **Reliability:** The inclusion of "Fail-Safe" token limits and strict vocab masks makes the system robust against edge-case inputs (e.g., negative numbers, nested quotes).

## Challenges Faced

* **Environment Desync:** Initially, `mypy` could not resolve the `pydantic` plugin, which was solved by explicitly managing `dependency-groups` in `pyproject.toml` and ensuring `uv sync --group dev` was executed.
* **Tokenization Limits:** Preventing infinite loops in the LLM required implementing hard token caps and `current_token_index` limiters in the FSM router.
* **Tensor vs Array Indexing:** During logit masking, PyTorch Tensors returned by the llm_sdk conflicted with NumPy boolean indexing arrays. We resolved this by explicitly casting np.array(raw_logits) to safely bridge the two data formats without throwing scalar index errors.
* **LLM "Babbling":** The LLM struggled with complex strings like SQL queries and Regex patterns, occasionally forgetting to output closing quotes. We implemented strict grammar termination delimiters and token generation safety caps to prevent CPU hanging.

## Testing Strategy

We utilized two JSON-based test suites:

1. `data/input/edge_complex_tests_new.json`: Targeted at adversarial scenarios (scientific notation, complex string escaping, negative numbers).
2. `data/input/edge_error_tests.json`: Targeted at structural edge cases (empty strings, nested JSON, malicious SQL injection attempts).

## Example Usage

**Request:** *"What is the sum of 2 and 3?"*
**Output:**

```json
{
    "id": "test_0",
    "prompt": "What is the sum of 2 and 3?",
    "result": {
      "function": "fn_add_numbers",
      "arguments": {
        "a": 2,
        "b": 3
      }
    }
  }

```

## Resources

* **Reference:** [Tiny LLMs and Agents on Edge Devices with LiteRT-LM](http://googleusercontent.com/youtube_content/0)
* **AI Usage:** Considering the novalty and advancement of the subject, AI was used extensively for analysis, planning, implementation, debugging, testing, validation and optimizing the logit masking implementation in `src/vocab.py`, as well as drafting this documentation.
