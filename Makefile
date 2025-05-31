.PHONY: requirements
requirements: export UV_CUSTOM_COMPILE_COMMAND='make requirements'
requirements:
	@uv pip compile --generate-hashes --strip-extras --extra=test --upgrade --output-file=requirements.txt pyproject.toml
