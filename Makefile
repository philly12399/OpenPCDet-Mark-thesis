.PHONY: build clean

ENPTY_TARGETS = .install_torch .install_pcdet .build_poetry

default: build

build: .install_torch .install_pcdet .build_poetry

.build_poetry:
	poetry install
	touch .build_poetry

.install_torch: .build_poetry
	poetry run \
	pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
	touch .install_torch

.install_pcdet: .install_torch
	poetry run python setup.py develop
	touch .install_pcdet

clean:
	rm -rf $(ENPTY_TARGETS)
	rm -rf build
	rm -f ./poetry.lock
	poetry env list | cut -f1 -d' ' | while read name; do \
		poetry env remove "$$name"; \
	done
