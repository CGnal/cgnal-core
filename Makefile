# Signifies our desired python version
# Makefile macros (or variables) are defined a little bit differently than traditional bash, keep in mind that in the Makefile there's top-level Makefile-only syntax, and everything else is bash script syntax.
PYTHON = python

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = help

folders := cgnal/core tests
files := $(shell find . -name "*.py")
doc_files := $(shell find sphinx -name "*.*")

# Uncomment to store cache installation in the environment
# package_dir := $(shell python -c 'import site; print(site.getsitepackages()[0])')
package_dir := .make_cache
package_name=$(shell python -c "import configparser;config=configparser.ConfigParser();config.read('setup.cfg');print(config['metadata']['name'])")

$(shell mkdir -p $(package_dir))

pre_deps_tag := $(package_dir)/.pre_deps
env_tag := $(package_dir)/.env_tag
env_ci_tag := $(package_dir)/.env_ci_tag
install_tag := $(package_dir)/.install_tag

# ======================
# Rules and Dependencies
# ======================

help:
	@echo "---------------HELP-----------------"
	@echo "Package Name: $(package_name)"
	@echo " "
	@echo "Type 'make' followed by one of these keywords:"
	@echo " "
	@echo "  - setup for installing base requirements"
	@echo "  - setup_ci for installing requirements for development"
	@echo "  - format for reformatting files to adhere to PEP8 standards"
	@echo "  - dist for building a tar.gz distribution"
	@echo "  - install for installing the package"
	@echo "  - uninstall for uninstalling the environment"
	@echo "  - tests for running unittests"
	@echo "  - lint for performing linting"
	@echo "  - mypy for performing static typing checks"
	@echo "  - docs for producing self-documentation"
	@echo "  - checks for running format, mypy, lint and tests altogether"
	@echo "  - clean for removing cache file"
	@echo "------------------------------------"

$(pre_deps_tag):
	grep "^pip-tools\|^black"  requirements/requirements_ci.in | xargs ${PYTHON} -m pip install
	touch $(pre_deps_tag)

requirements/requirements.txt: requirements/requirements_ci.txt
	cat requirements/requirements.in > subset.in
	echo "\n-c requirements/requirements_ci.txt" >> subset.in
	pip-compile --output-file "requirements/requirements.txt" --quiet --no-emit-index-url subset.in
	rm subset.in

reqs: requirements/requirements.txt

requirements/requirements_ci.txt: requirements/requirements_ci.in requirements/requirements.in $(pre_deps_tag)
	pip-compile --output-file requirements/requirements_ci.txt --quiet --no-emit-index-url requirements/requirements.in requirements/requirements_ci.in

reqs_ci: requirements/requirements_ci.txt

$(env_tag): requirements/requirements.txt
	pip-sync requirements/requirements.txt
	rm -f $(env_ci_tag)
	touch $(env_tag)

$(env_ci_tag): requirements/requirements_ci.txt
	pip-sync requirements/requirements_ci.txt
	rm -f $(env_tag)
	touch $(env_ci_tag)

setup: $(env_tag)

setup_ci: $(env_ci_tag)

format: setup_ci
	${PYTHON} -m black $(folders)

dist/.build-tag: $(files) setup.cfg requirements/requirements.txt
	${PYTHON} setup.py sdist
	ls -rt  dist/* | tail -1 > dist/.build-tag

dist: dist/.build-tag setup.py

$(install_tag): dist/.build-tag
	${PYTHON} -m pip install $(shell ls -rt  dist/* | tail -1)
	touch $(install_tag)

uninstall:
	@echo "Uninstall package $(package_name)"
	pip uninstall -y $(package_name)
	pip freeze | grep -v "@" | xargs pip uninstall -y
	rm -f $(env_tag) $(env_ci_tag) $(pre_deps_tag) $(install_tag)

install: setup $(install_tag)

tests: $(install_tag) setup_ci
	${PYTHON} -m pytest

mypy: $(install_tag) setup_ci
	mypy --install-types --non-interactive --follow-imports silent $(folders)

lint: setup_ci
	flake8 $(folders)

checks: format mypy lint tests

docs: setup_ci $(install_tag) $(doc_files) setup.cfg
	sphinx-apidoc -f -o sphinx/source/api cgnal/core
	make --directory=sphinx --file=Makefile clean html

clean: uninstall
	rm -rf docs
	rm -rf build
	rm -rf $(shell find . -name "*.pyc") $(shell find . -name "__pycache__")
	rm -rf dist *.egg-info .mypy_cache .pytest_cache .make_cache $(env_tag) $(env_ci_tag) $(install_tag)
