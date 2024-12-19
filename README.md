### philosophy

Goal: easy to hack around.

- Minimal implementations and imports. Keep everything modular. 
- Keep everything as easy as possible to see inputs/outputs. 
- Uniform.
- No need to go down folder hierarchy to track bugs. 


### workflow
1. safety_data: `jsonl` files for prompting.
2. safety_data_configs: `toml` files for prompting with `vllm`.
3. `generate.py`: take (1, 2) and generate model outputs.
4. `evaluate.py`: classify (2)