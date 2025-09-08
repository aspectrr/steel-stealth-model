# produce sft_train.jsonl with records like:
# {"instruction": "...", "input": "...", "output": "..."}
def build_jsonl(records, out_path="sft_train.jsonl"):
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps({"instruction": r.inst, "input": r.input, "output": r.output}) + "\n")
