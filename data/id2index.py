import os
import json

KEYFRAMES_DIR = "data/keyframes"
OUTPUT_JSON = "data/id2index.json"

id2index = {}
current_id = 0

# Duyệt từng Lxx
for L_dir in sorted(os.listdir(KEYFRAMES_DIR)):
    L_path = os.path.join(KEYFRAMES_DIR, L_dir)
    if not os.path.isdir(L_path):
        continue

    # Lấy số L (vd "L23" -> 23)
    L_num = int(L_dir.replace("L", ""))

    # Duyệt từng Vxxx
    for V_dir in sorted(os.listdir(L_path)):
        V_path = os.path.join(L_path, V_dir)
        if not os.path.isdir(V_path):
            continue

        # Lấy số V (vd "L23_V001" -> 1)
        V_num = int(V_dir.split("_V")[1])

        # Duyệt từng frame ảnh
        for frame_file in sorted(os.listdir(V_path)):
            if frame_file.lower().endswith((".jpg", ".png", ".webp")):
                frame_idx = int(os.path.splitext(frame_file)[0])
                id2index[str(current_id)] = f"{L_num}/{V_num}/{frame_idx}"
                current_id += 1

# Lưu ra file
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(id2index, f, ensure_ascii=False, indent=2)

print(f"[DONE] Saved {len(id2index)} mappings to {OUTPUT_JSON}")
