.PHONY: test test-registry test-config test-all

test-all:
	python -m pytest tests/

test-registry:
	python -m pytest -v tests/test_registry.py 

test-config:
	python -m pytest -v tests/test_config.py 

test-builder:
	python -m pytest -v tests/test_builder.py

test-loss-functions:
	python -m pytest -v tests/test_loss_functions.py

test-regression-losses:
	python -m pytest -v tests/test_regression_losses.py

test-lightning-detector:
	python -m pytest -v tests/test_lightning_detector.py

test-anchor-generator:
	python -m pytest -v tests/test_anchor_generator.py

test-fpn-image-feature-extractor:
	python -m pytest -v tests/test_fpn_image_feature_extractor.py

test-sparse4d-loss:
	python -m pytest -v tests/test_sparse4d_loss.py

test-coverage:
	python -m pytest -v --cov=xinnovation tests/