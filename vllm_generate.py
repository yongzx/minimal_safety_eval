from vllm import LLM, SamplingParams
import argparse
import pathlib
from loguru import logger
import toml
import json

DATA_NAMES = [
    fp.stem for fp in pathlib.Path("./safety_data/").glob("*.jsonl")
]

logger.info(f"{DATA_NAMES=}")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use for generation")
parser.add_argument("--data_name", choices=DATA_NAMES, default="xstest_v2_prompts", help="Name of dataset to use")
parser.add_argument("--root_dir", default=".")
parser.add_argument("--output_dir", default="./model_outputs")
args = parser.parse_args()
root_dir = pathlib.Path(args.root_dir)
output_dir = pathlib.Path(args.output_dir)

##### load VLLM generation config (benchmark-specific)
with open(root_dir / "safety_data_configs" / f"{args.data_name}.toml") as rf:
    config = toml.load(rf)
logger.info(f"Loaded data {config=}")

####### load dataset
data = dict() # input_prompt -> e
with open(root_dir / "safety_data" / f"{args.data_name}.jsonl") as rf:
    for line in rf:
        line = json.loads(line)
        data[line["input_prompt"]] = line

match args.data_name:
    case "sorry_bench_202406":
        prompts = list(data.keys())
    case "aya_m_prism":
        prompts = list(data.keys())
    case "xstest_v2_prompts":
        prompts = list(data.keys())
    case _:
        prompts = []

sampling_params = SamplingParams(**config["vllm_config"])
llm = LLM(model=args.model)
outputs = llm.generate(prompts, sampling_params)

jsonl_wf = open(root_dir / "model_outputs" / f"data::{args.data_name}-model::{args.model.split('/')[-1]}.jsonl", "w+")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    jsonl_wf.write(json.dumps({
        "output": generated_text,
        **data[prompt],
    }) + "\n")

