import pandas as pd
import json

df = pd.read_csv("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/raw/sorry_bench_202406.csv")

with open("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/sorry_bench_202406.jsonl", "w+") as wf:
    for index, row in df.iterrows():
        wf.write(json.dumps({
            "input_prompt": row["prompt"],
            "category": row["category"]
        }) + "\n")