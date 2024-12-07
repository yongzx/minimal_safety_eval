import pathlib
import json

wf = open("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/lrl_jailbreak_gpt4.jsonl", "w+")
for fp in pathlib.Path("/users/zyong2/data/zyong2/minimal_safety_eval/safety_data/raw/").glob("zou*.jsonl"):
    with open(fp) as rf:
        for line in rf:
            line = json.loads(line)
            wf.write(json.dumps({
                "input_prompt": line["translatedText"],
                "en_prompt": line["input"],
                "lang": fp.stem.split("_")[-1],
            }) + '\n')

wf.close()