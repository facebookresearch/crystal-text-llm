"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

conda create -n crystal-llm
conda activate crystal-llm

pip install torch torchvision torchaudio
pip install accelerate peft transformers sentencepiece datasets
pip install bitsandbytes

conda deactivate
