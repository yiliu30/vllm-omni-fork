import json
import os
import sys

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/.cache/modelscope/hub/models/asdazd/Qwen3-Omni-30B-A3B-Instruct_modelopt_FP8"

idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
shards = set(idx["weight_map"].values())
missing = [s for s in shards if not os.path.exists(os.path.join(model_dir, s))]
total_gb = sum(
    os.path.getsize(os.path.join(model_dir, s))
    for s in shards
    if os.path.exists(os.path.join(model_dir, s))
) / 1e9

print(f"Shards: {len(shards)}, Missing: {len(missing)}, Size: {total_gb:.1f} GB")
if missing:
    print("Missing:", missing)
else:
    print("Download complete")
