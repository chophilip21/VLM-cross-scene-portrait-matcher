install:
	pip install --upgrade pip
	pip install --upgrade -e .[devel]

install-production:
	pip install --upgrade pip
	pip install --upgrade -e .

build:
	python3 -m pip install --upgrade build
	python3 -m build

lint:
	flake8 src tests

docstyle:
	pydocstyle -v src tests

isort:
	isort --verbose --check-only --diff src tests

black:
	black --verbose --check --diff src tests

fix-code:
	isort --verbose src tests
	black --verbose src tests
	
dvc-push:
	helpers/data_versioning/push.sh

dvc-pull:
	helpers/data_versioning/pull.sh

package-models:
	helpers/export/export_models.sh

build-engine:
	helpers/engine/build_torchserve.sh

launch-engine:
	$(MAKE) build-engine
	helpers/engine/launch_torchserve.sh

run-compose:
	helpers/compose/run_docker_compose.sh

stop-compose:
	helpers/compose/stop_docker.sh

run-worker:
	helpers/worker/launch_worker.sh

build-production:
	$(MAKE) build-engine
	$(MAKE) run-compose
	$(MAKE) run-worker