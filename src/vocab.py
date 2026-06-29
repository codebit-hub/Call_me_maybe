import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


# Censor: Numpy Boolean Mask Generator.
# When the application initializes, this module will instantiate the LLM,
# read its vocabulary size ($N = 151,936$), and decode every token ID
# from 0 to 151,935 into text. It categorizes them into static C-level
# boolean arrays of size (151,936). If mask[token_id] == True,
# the LLM is permitted to speak it. If False, our state machine
# will set its logit score to negative infinity -inf.


class VocabMasks:
    """Pre-computes static C-level boolean masks for token constrains"""

    def __init__(self, llm: Small_LLM_Model, vocab_size: int = 151936) -> None:
        """Instantiates mask arrays and calls their compilation"""

        self.vocab_size: int = vocab_size

        # Allocates contiguous memory tables (default False / Banned)
        # Initiates and sets tables to 0
        self.digits: np.ndarray = np.zeros(vocab_size, dtype=bool)
        self.booleans: np.ndarray = np.zeros(vocab_size, dtype=bool)
        self.nulls: np.ndarray = np.zeros(vocab_size, dtype=bool)

        # Atomic structural grammar masks
        # Initiates and sets masks to 0
        self.open_brace: np.ndarray = np.zeros(vocab_size, dtype=bool)
        self.close_brace: np.ndarray = np.zeros(vocab_size, dtype=bool)
        self.colon: np.ndarray = np.zeros(vocab_size, dtype=bool)
        self.comma: np.ndarray = np.zeros(vocab_size, dtype=bool)
        self.quote: np.ndarray = np.zeros(vocab_size, dtype=bool)

        self._compile_masks(llm)

    def _compile_masks(self, llm: Small_LLM_Model) -> None:
        """Iterates the vocab once to populate binary truth tables"""

        # Contains digits and signs: +, -, e, E, float . ,
        valid_number_chars: set[str] = set("0123456789+-.eE")

        for token_id in range(self.vocab_size):
            try:
                raw_str: str = llm.decode([token_id])
                # Retrieves and decodes every token from LLM
                # which is passed on to validation
                # and is stored in respective ndarrays
                # created ndarrays match llm reponse vocab_size
                # this enables fast biwise ndarrays matching

            except (ValueError, UnicodeDecodeError):
                # This captures only relevant errors
                # Skips partial byte fragments
                continue

            s: str = raw_str.strip()
            if not s:
                # Pure whitespace tokens are are to be skipped
                continue

            # 1. Numeric components
            if all(c in valid_number_chars for c in s):
                self.digits[token_id] = True

            # 2. Booleans
            if s in ("true", "false"):
                self.booleans[token_id] = True

            # 3. Null
            if s == "null":
                self.nulls[token_id] = True

            # 4. Structural JSON delimiters
            if s == "{":
                self.open_brace[token_id] = True
            elif s == "}":
                self.close_brace[token_id] = True
            elif s == ":":
                self.colon[token_id] = True
            elif s == ",":
                self.comma[token_id] = True
            elif s == '"':
                self.quote[token_id] = True


if __name__ == "__main__":
    import sys
    import time

    print("Initializing Small_LLM_Model on CPU...")
    t0 = time.time()
    # t0: 1782385373.1547275 sec

    try:
        model = Small_LLM_Model(device="cpu")
        load_time = time.time() - t0
        print(f"Model bound to HW in {load_time:.2f} seconds.")

        print("Compiling vocab truth tables...")
        t1 = time.time()
        masks = VocabMasks(model)
        compile_time = time.time() - t1

        print(f"[SUCCESS]: Processed 151,936 tokens in {compile_time:.2f}s.\n")
        print(f" ├─ Valid Numeric Tokens:    {np.sum(masks.digits)}")
        print(f" ├─ Valid Boolean Tokens:    {np.sum(masks.booleans)}")
        print(f" ├─ Open Brace ('{{') Tokens: {np.sum(masks.open_brace)}")
        print(f" └─ Quote ('\"') Tokens:      {np.sum(masks.quote)}")

    except Exception as err:
        print(f"\n[FATAL ERROR] Vocabulary compilation failed: {err}")
        sys.exit(1)
