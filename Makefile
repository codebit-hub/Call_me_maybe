# Constrained Decoding Agent

NAME	= constrained_decoder

# Toolchain
PYTHON	= uv run python
FLAKE8	= uv run flake8
MYPY	= uv run mypy

# Directories
SRC_DIR		= src
DATA_DIR	= data

# Default target
.PHONY: all
all: run

# Execution Rules

.PHONY: run
run:
	@echo "Running Constrained Decoding Agent (FSM ON)..."
	$(PYTHON) -m $(SRC_DIR)

.PHONY: run_off
run_off:
	@echo "Running Unconstrained Decoding Agent (FSM OFF)..."
	$(PYTHON) -m $(SRC_DIR) --off

# Validation Rules

.PHONY: lint
lint:
	@echo "Running Flake8 (Formatting)..."
	$(FLAKE8) $(SRC_DIR)
	@echo "Running MyPy (Static Type Checking)..."
	$(MYPY) $(SRC_DIR) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

# Cleanup Rules

.PHONY: clean
clean:
	@echo "Cleaning cache files..."
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: fclean
fclean: clean
	@echo "Performing deep clean..."
	rm -rf .venv

.PHONY: re
re: fclean all
