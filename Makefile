.PHONY: requirements
requirements: export CUSTOM_COMPILE_COMMAND='make requirements'
requirements:
	@uv pip compile --generate-hashes --strip-extras --extra=test --upgrade --output-file=requirements.txt pyproject.toml
