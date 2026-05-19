"""
将 backend/data 目录分卷压缩，确保每个zip文件不超过90MB。
对于超过90MB的单个文件，将其拆分存储到多个zip中。
解压方式：运行 extract_data.py 即可自动合并还原。
"""
import os
import zipfile
import sys
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data_archives")
MAX_ZIP_SIZE = 90 * 1024 * 1024  # 90MB
CHUNK_SIZE = 80 * 1024 * 1024  # 80MB per chunk for large files


def split_zip():
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            os.remove(fp)

    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            relpath = os.path.relpath(filepath, DATA_DIR)
            all_files.append((filepath, relpath))

    all_files.sort(key=lambda x: os.path.getsize(x[0]), reverse=True)

    total_size = sum(os.path.getsize(f[0]) for f in all_files)
    print(f"共 {len(all_files)} 个文件，总大小 {total_size / 1024 / 1024:.1f} MB")

    manifest = []
    vol_idx = 0
    vol_zip = None
    vol_size = 0

    def open_new_vol():
        nonlocal vol_idx, vol_size, vol_zip
        if vol_zip is not None:
            vol_zip.close()
        vol_idx += 1
        vol_size = 0
        path = os.path.join(OUTPUT_DIR, f"data_part{vol_idx:02d}.zip")
        print(f"  创建分卷 {vol_idx:02d}")
        vol_zip = zipfile.ZipFile(path, "w", zipfile.ZIP_STORED)
        return path

    for filepath, relpath in all_files:
        fsize = os.path.getsize(filepath)

        if fsize <= CHUNK_SIZE:
            if vol_zip is None or vol_size + fsize > MAX_ZIP_SIZE:
                open_new_vol()
            vol_zip.write(filepath, relpath)
            vol_size += fsize
            manifest.append({"path": relpath, "split": False, "volumes": [vol_idx]})
            if fsize > 1 * 1024 * 1024:
                print(f"  + {relpath} ({fsize / 1024 / 1024:.1f} MB) -> part{vol_idx:02d}")
        else:
            chunk_idx = 0
            remaining = fsize
            vol_list = []
            with open(filepath, "rb") as fin:
                while remaining > 0:
                    read_size = min(CHUNK_SIZE, remaining)
                    data = fin.read(read_size)
                    if vol_zip is None or vol_size + len(data) > MAX_ZIP_SIZE:
                        open_new_vol()
                    chunk_name = f"{relpath}.part{chunk_idx:03d}"
                    vol_zip.writestr(chunk_name, data)
                    vol_size += len(data)
                    vol_list.append(vol_idx)
                    remaining -= len(data)
                    chunk_idx += 1
            manifest.append({"path": relpath, "split": True, "chunks": chunk_idx, "volumes": vol_list})
            print(f"  + {relpath} ({fsize / 1024 / 1024:.1f} MB) -> {chunk_idx} chunks across parts {vol_list}")

    if vol_zip is not None:
        vol_zip.close()

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n分卷压缩完成！共 {vol_idx} 个分卷")
    print(f"清单文件: {manifest_path}")
    print(f"保存位置: {OUTPUT_DIR}")


if __name__ == "__main__":
    split_zip()
