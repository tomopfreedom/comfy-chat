"""WD14 selected_tags.csv + 日本語翻訳 → static/tags.json 変換スクリプト。

翻訳データ優先順位:
  1. danbooru-jp.csv      (人力翻訳, 427件)
  2. danbooru-machine-jp.csv (機械翻訳, ~100k件)
"""
import csv, json, pathlib

TOOLS = pathlib.Path(__file__).parent
SRC   = pathlib.Path("/home/tomop/infra/kohya_ss/wd14_tagger_model"
                     "/SmilingWolf_wd-v1-4-convnextv2-tagger-v2/selected_tags.csv")
DST   = TOOLS.parent / "static" / "tags.json"

def load_translations(path):
    t = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[1].strip():
                    t[row[0].strip()] = row[1].strip()
    return t

# 機械翻訳を読み込み、人力翻訳でオーバーライド
trans = load_translations(TOOLS / "danbooru-machine-jp.csv")
trans.update(load_translations(TOOLS / "danbooru-jp.csv"))
print(f"Translations loaded: {len(trans)}")

tags = []
with open(SRC, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["category"]) != 0:
            continue
        entry = {"n": row["name"], "s": int(row["count"])}
        ja = trans.get(row["name"])
        if ja:
            entry["j"] = ja
        tags.append(entry)

tags.sort(key=lambda x: -x["s"])

DST.write_text(json.dumps(tags, separators=(",", ":")), encoding="utf-8")
covered = sum(1 for t in tags if "j" in t)
print(f"Written {len(tags)} tags ({covered} with Japanese) → {DST}  ({DST.stat().st_size // 1024} KB)")
