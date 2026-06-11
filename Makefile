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

# Réentraînement diagnostics RL (§6.3.2) -- scénarios high_load / high_breakdown
# 1 thread/job (OMP_NUM_THREADS=1 MKL_NUM_THREADS=1) pour que les 8 jobs
# tiennent sur 8 coeurs sans sur-sollicitation. PYTHONIOENCODING=utf-8 est
# necessaire pour Q-Learning/SARSA (caractere "epsilon" dans les print()).

.PHONY: train-ql-highload
train-ql-highload:
	set "PYTHONIOENCODING=utf-8" && set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_diagnostics q_learning high_load > data/results/qlearning_diag_highload.log 2>&1

.PHONY: train-sarsa-highload
train-sarsa-highload:
	set "PYTHONIOENCODING=utf-8" && set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_diagnostics sarsa high_load > data/results/sarsa_diag_highload.log 2>&1

.PHONY: train-ql-highbreakdown
train-ql-highbreakdown:
	set "PYTHONIOENCODING=utf-8" && set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_diagnostics q_learning high_breakdown > data/results/qlearning_diag_highbreakdown.log 2>&1

.PHONY: train-sarsa-highbreakdown
train-sarsa-highbreakdown:
	set "PYTHONIOENCODING=utf-8" && set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_diagnostics sarsa high_breakdown > data/results/sarsa_diag_highbreakdown.log 2>&1

.PHONY: train-dqn-highload
train-dqn-highload:
	set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_losscurve dqn high_load > data/results/dqn_losscurve_highload.log 2>&1

.PHONY: train-ppo-highload
train-ppo-highload:
	set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_losscurve ppo high_load > data/results/ppo_losscurve_highload.log 2>&1

.PHONY: train-dqn-highbreakdown
train-dqn-highbreakdown:
	set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_losscurve dqn high_breakdown > data/results/dqn_losscurve_highbreakdown.log 2>&1

.PHONY: train-ppo-highbreakdown
train-ppo-highbreakdown:
	set "OMP_NUM_THREADS=1" && set "MKL_NUM_THREADS=1" && $(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.train_losscurve ppo high_breakdown > data/results/ppo_losscurve_highbreakdown.log 2>&1

## Lancer les 8 reentrainements diagnostics RL en parallele (high_load + high_breakdown)
.PHONY: train-diagnostics-all
train-diagnostics-all:
	$(MAKE) -j8 train-ql-highload train-sarsa-highload train-ql-highbreakdown train-sarsa-highbreakdown train-dqn-highload train-ppo-highload train-dqn-highbreakdown train-ppo-highbreakdown

## Generer les figures TD-error / Policy-Value Loss / Exploration (3 scenarios)
.PHONY: plot-diagnostics
plot-diagnostics:
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.plot_td_error
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.plot_policy_value_loss
	$(PYTHON_INTERPRETER) -m memoire_master_rl_logistique.experiments.plot_exploration_exploitation


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
