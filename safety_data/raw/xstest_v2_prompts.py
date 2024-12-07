import pandas as pd
import json

df = pd.read_csv("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/raw/xstest_v2_prompts.csv")

with open("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/xstest_v2_prompts.jsonl", "w+") as wf:
    for index, row in df.iterrows():
        wf.write(json.dumps({
            "input_prompt": row["prompt"],
            "type": row["type"],
            "focus": row["focus"],
            "note": row["note"],
        }) + "\n")
