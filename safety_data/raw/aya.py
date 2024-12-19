import pathlib
import json

wf = open("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/aya_m_prism.jsonl", "w+")
for fp in pathlib.Path("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/raw/").glob("aya_*.jsonl"):
    with open(fp) as rf:
        for line in rf:
            line = json.loads(line)
            line["input_prompt"] = line["prompt"]
            del line["prompt"]
            wf.write(json.dumps(line) + '\n')

wf.close()