import os, json
import numpy as np
import torch
from pathlib import Path

ID2INDEX = Path("data/id2index.json")
FEATURE_DIR = Path("data/features")
OUT_PT = Path("data/embeddings.pt")


def npy_path(group_num: int, video_num: int) -> Path:
    return FEATURE_DIR / f"L{group_num:02d}_V{video_num:03d}.npy"


def main():
    id2idx = json.load(open(ID2INDEX, "r", encoding="utf-8"))
    N = len(id2idx)

    embeddings = []
    last_file = None
    arr = None

    for i in range(N):
        g, v, k = map(int, id2idx[str(i)].split("/"))
        p = npy_path(g, v)

        if p != last_file:
            if not p.exists():
                raise FileNotFoundError(f"Missing feature file: {p}")
            arr = np.load(p)  # expected shape [num_frames, D]
            if arr.ndim != 2:
                raise ValueError(f"Bad shape for {p}: {arr.shape}")
            last_file = p

        # keyframe_num starts at 1 → index = keyframe_num - 1
        idx = k - 1
        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(
                f"Frame index {idx} out of range for {p} with {arr.shape[0]} frames"
            )

        embeddings.append(arr[idx])

    mat = np.stack(embeddings, axis=0).astype("float32")
    print(f"Final matrix shape: {mat.shape}")  # [N, D]

    torch.save(torch.from_numpy(mat), OUT_PT)
    print(f"Saved embeddings to {OUT_PT}")


if __name__ == "__main__":
    main()
