default: style-fix style-check

style-fix: python-style-fix
style-check: python-style-check

IGNORE = lcb_runner
PYTHON_FILES:=$(wildcard *.py)
PYTHON_FILES_TO_CHECK:=$(filter-out ${lcb_runner},${PYTHON_FILES})
install-mypy:
	@if ! command -v mypy ; then pip install mypy ; fi
install-ruff:
	@if ! command -v ruff ; then pipx install ruff ; fi
python-style-fix: install-ruff
	@ruff format ${PYTHON_FILES_TO_CHECK}
	@ruff -q check ${PYTHON_FILES_TO_CHECK} --fix
python-style-check: install-ruff
	@ruff -q format --check ${PYTHON_FILES_TO_CHECK}
	@ruff -q check ${PYTHON_FILES_TO_CHECK}
python-typecheck: install-mypy
	@mypy --strict ${PYTHON_FILES_TO_CHECK} > /dev/null 2>&1 || true
	@mypy --install-types --non-interactive
	mypy --strict --ignore-missing-imports ${PYTHON_FILES_TO_CHECK}


showvars:
	echo "PYTHON_FILES=${PYTHON_FILES}"
	echo "PYTHON_FILES_TO_CHECK=${PYTHON_FILES_TO_CHECK}"
