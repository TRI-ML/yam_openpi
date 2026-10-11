#!/bin/bash
# Assemble the mt9 fine-tune repos from build_mt9.py sources: train = all success episodes minus the 5%/task holdout,
# val = the holdout. Then per-repo lowdim memmaps (the loader reads them when LEROBOT_LOWDIM_MEMMAP=1, the default).
# usage: SOURCES=/tmp/mt9/sources HF_LEROBOT_HOME=/tmp/mt9/lerobot_home bash yam/finetune/assemble_mt9.sh [train_repo] [val_repo]
set -euo pipefail
SOURCES=${SOURCES:?}; HF_LEROBOT_HOME=${HF_LEROBOT_HOME:?}; TRAIN=${1:-local/yam_mt9_v21}; VAL=${2:-local/yam_mt9_val_v21}
PY=${PY:-.venv/bin/python}
tasks=$(ls -d "$SOURCES"/*/ | xargs -n1 basename | grep -v '^summary' )
excl=(); only=()
for t in $tasks; do excl+=("$t:$SOURCES/$t/val_ids.txt"); only+=("$t:$SOURCES/$t/val_ids.txt"); done
srcs=(); for t in $tasks; do srcs+=("$SOURCES/$t"); done
$PY yam/pretrain/assemble_v21.py "$HF_LEROBOT_HOME/$TRAIN" "${srcs[@]}" --exclude "${excl[@]}" --workers 16
$PY yam/pretrain/assemble_v21.py "$HF_LEROBOT_HOME/$VAL"   "${srcs[@]}" --only    "${only[@]}" --workers 16
# the assembler stamps the pretrain's AV1 256x416 video feature; ours is h264 at the size build_mt9 used -> fix info.json
for r in "$TRAIN" "$VAL"; do $PY - "$HF_LEROBOT_HOME/$r" "$SOURCES" <<'PYEOF'
import json, sys, glob
root, S = sys.argv[1], sys.argv[2]
src = json.load(open(glob.glob(f"{S}/*/source.json")[0]))["video"]; w, h = (int(x) for x in src["size"].split("x"))
info = json.load(open(f"{root}/meta/info.json"))
for k, v in info["features"].items():
    if v["dtype"] == "video":
        v["shape"] = [h, w, 3]; v["info"].update({"video.height": h, "video.width": w, "video.codec": "h264", "note": f"build_mt9.py re-encode of the 1280x720 station videos to {w}x{h} (aspect kept); trainer resizes to 224x224"})
json.dump(info, open(f"{root}/meta/info.json", "w"), indent=2); print("fixed video feature ->", [h, w, 3], root)
PYEOF
done
for r in "$TRAIN" "$VAL"; do $PY yam/pretrain/build_lowdim_memmap.py "$HF_LEROBOT_HOME/$r" --workers 16; done
