# Constrained Decoding Agent

NAME	= constrained_decoder
SHELL	= /bin/bash

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

# Installation Rules

.PHONY: install
install:
	@if [[ ! -f .install_stamp || pyproject.toml -nt .install_stamp ]]; then \
		echo "Installing project dependencies using uv..."; \
		uv sync --all-groups; \
		touch .install_stamp; \
	else \
		echo "Nothing to do: everything is already setup."; \
	fi

# Execution Rules

.PHONY: run
run: install
	@echo "Running Constrained Decoding Agent (FSM ON)..."
	$(PYTHON) -m $(SRC_DIR)

.PHONY: run_off
run_off: install
	@echo "Running Unconstrained Decoding Agent (FSM OFF)..."
	$(PYTHON) -m $(SRC_DIR) --off

# Debugging Mode

.PHONY: debug
debug: install
	@echo "Running Constrained Decoding Agent in Debug Mode..."
	$(PYTHON) -m pdb -m $(SRC_DIR)

# Validation Rules

.PHONY: lint
lint:
	@echo "Running Flake8 (Formatting)..."
	$(FLAKE8) $(SRC_DIR)
	@echo "Flake8 check completed."
	@echo "Running MyPy (Static Type Checking)..."
	$(MYPY) $(SRC_DIR) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

.PHONY: lint-strict
lint-strict:
	@echo "Running Flake8 (Formatting)..."
	$(FLAKE8) $(SRC_DIR)
	@echo "Flake8 check completed."
	@echo "Running MyPy (Strict Static Type Checking)..."
	$(MYPY) $(SRC_DIR) --strict

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
	rm -f .install_stamp

.PHONY: re
re: fclean all
