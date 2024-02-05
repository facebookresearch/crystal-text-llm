conda create -n crystal-llm
conda activate crystal-llm

pip install torch torchvision torchaudio
pip install accelerate peft transformers sentencepiece datasets
pip install bitsandbytes

conda deactivate
