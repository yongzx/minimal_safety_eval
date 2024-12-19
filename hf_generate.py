import argparse
import pathlib
from loguru import logger
import toml
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

DATA_NAMES = [
    fp.stem for fp in pathlib.Path("./safety_data/").glob("*.jsonl")
]

logger.info(f"{DATA_NAMES=}")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct", help="Model to use for generation")
parser.add_argument("--data_name", choices=DATA_NAMES, default="sorry_bench_202406", help="Name of dataset to use")
parser.add_argument("--root_dir", default=".")
parser.add_argument("--output_dir", default="./model_outputs")
parser.add_argument("--bsz", type=int, default=32)
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

### generation
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")

jsonl_wf = open(root_dir / "model_outputs" / f"data::{args.data_name}-model::{args.model.split('/')[-1]}.jsonl", "w+")
for i in tqdm(range(0, len(prompts)//args.bsz + 1)):
    subset_prompts = prompts[i*args.bsz:(i+1)*args.bsz]
    if not subset_prompts:
        break
    model_inputs = tokenizer(subset_prompts, return_tensors="pt", padding=True).to("cuda") #TODO: if OOM, perform batch inference.
    generated_ids = model.generate(
        **model_inputs,
        temperature=config["vllm_config"]["temperature"],
        top_p=config["vllm_config"]["top_p"],
        max_new_tokens=256,
        do_sample=True,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for j, output in enumerate(outputs):
        prompt = subset_prompts[j]
        generated_text = output[len(prompt):]
        jsonl_wf.write(json.dumps({
            "output": generated_text,
            **data[prompt],
        }) + "\n")




# print()
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text


