### philosophy

Goal: easy to hack around.

- Minimal implementations and imports. Keep everything modular. 
- Keep everything as easy as possible to see inputs/outputs. 
- Uniform.
- No need to go down folder hierarchy to track bugs. 


### data
1. safety_data: `jsonl` files
2. model_outputs: `jsonl` files containing model outputs.
3. `generate.py`: take (1) and output to (2)
4. `evaluate.py`: classify (2)