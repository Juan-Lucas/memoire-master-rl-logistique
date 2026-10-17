#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = memoire-master-rl-logistique
PYTHON_VERSION = 3.11.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m unittest discover -s tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# Figures d'entraînement (T30) : courbes moyennes ± IC 95 % sur les graines d'entraînement,
# lues dans les journaux des modèles évalués par la campagne (plus de réentraînement séparé).
# Les scripts refusent d'écraser une figure existante (jamais reports/figures, article v1).
RESULTS ?= data/results_v2
FIGURES ?= reports/figures_v2

## Courbes de convergence, pertes, erreur TD et exploration des modèles de la campagne
.PHONY: training-figures
training-figures:
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.plots.convergence_qlearning_sarsa $(RESULTS) $(FIGURES)
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.plots.convergence_deeprl $(RESULTS) $(FIGURES)
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.plots.td_error $(RESULTS) $(FIGURES)
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.plots.policy_value_loss $(RESULTS) $(FIGURES)
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.plots.exploration_exploitation $(RESULTS) $(FIGURES)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
