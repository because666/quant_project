"""
解压分卷数据，还原 backend/data 目录。
用法：python extract_data.py
"""
import os
import zipfile
import json
import sys

ARCHIVES_DIR = os.path.join(os.path.dirname(__file__), "data_archives")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def extract():
    manifest_path = os.path.join(ARCHIVES_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"清单文件不存在: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    split_files = {}
    normal_vols = set()

    for entry in manifest:
        if entry["split"]:
            split_files[entry["path"]] = entry
        else:
            for v in entry["volumes"]:
                normal_vols.add(v)

    print(f"步骤1：解压普通文件（来自 {len(normal_vols)} 个分卷）")
    for vol in sorted(normal_vols):
        zip_path = os.path.join(ARCHIVES_DIR, f"data_part{vol:02d}.zip")
        if not os.path.exists(zip_path):
            print(f"  警告：分卷文件不存在: {zip_path}")
            continue
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if ".part" not in name:
                    target = os.path.join(OUTPUT_DIR, name)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        print(f"  已解压: part{vol:02d}.zip")

    print(f"步骤2：合并拆分文件（共 {len(split_files)} 个）")
    for relpath, entry in split_files.items():
        target = os.path.join(OUTPUT_DIR, relpath)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        chunk_data = {}
        for vol in entry["volumes"]:
            zip_path = os.path.join(ARCHIVES_DIR, f"data_part{vol:02d}.zip")
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if name.startswith(relpath + ".part"):
                        with zf.open(name) as src:
                            chunk_data[name] = src.read()

        with open(target, "wb") as dst:
            for i in range(entry["chunks"]):
                chunk_name = f"{relpath}.part{i:03d}"
                if chunk_name in chunk_data:
                    dst.write(chunk_data[chunk_name])
                else:
                    print(f"  警告：缺失分块 {chunk_name}")

        print(f"  已合并: {relpath} ({entry['chunks']} chunks)")

    print(f"\n解压完成！数据已还原到: {OUTPUT_DIR}")


if __name__ == "__main__":
    extract()
