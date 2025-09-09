import os, csv
from functools import lru_cache


def _csv_path(map_root: str, prefix: str, group_num: int, video_num: int) -> str:
    return os.path.join(map_root, f"{prefix}{group_num:02d}_V{video_num:03d}.csv")


@lru_cache(maxsize=1024)
def load_n2frame_idx(
    map_root: str, prefix: str, group_num: int, video_num: int
) -> dict[int, int]:
    """
    Trả về dict: n (1-based) -> frame_idx (int) cho video Lxx_Vyyy
    """
    path = _csv_path(map_root, prefix, group_num, video_num)
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # cột 'n' và 'frame_idx' theo format bạn cung cấp
            n = int(row["n"])
            frame_idx = int(float(row["frame_idx"]))
            table[n] = frame_idx
    return table


def n_to_frame_idx(
    map_root: str, prefix: str, group_num: int, video_num: int, n: int
) -> int | None:
    table = load_n2frame_idx(map_root, prefix, group_num, video_num)
    return table.get(n)
