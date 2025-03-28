.PHONY: test test-registry test-config test-all

test-all:
	python -m pytest tests/

test-registry:
	python -m pytest -v tests/test_registry.py 

test-config:
	python -m pytest -v tests/test_config.py 

test-builder:
	python -m pytest -v tests/test_builder.py

test-lightning-detector:
	python -m pytest -v tests/test_lightning_detector.py

test-loss-functions:
	python -m pytest -v tests/test_loss_functions.py

test-coverage:
	python -m pytest -v --cov=xinnovation tests/