import numpy as np
from PIL import Image

from metric_study.manifest import build_manifest, read_manifest, write_manifest


def _write_jpg(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path, quality=90)


def test_build_manifest_deduplicates_by_content_hash(tmp_path):
    arr_a = np.random.default_rng(0).uniform(0, 255, size=(32, 32))
    arr_b = np.random.default_rng(1).uniform(0, 255, size=(32, 32))
    _write_jpg(tmp_path / "1.jpg", arr_a)
    _write_jpg(tmp_path / "2.jpg", arr_a)  # different name, will differ slightly on re-save; force identical bytes:
    (tmp_path / "2.jpg").write_bytes((tmp_path / "1.jpg").read_bytes())
    _write_jpg(tmp_path / "3.jpg", arr_b)

    df = build_manifest(tmp_path)
    assert len(df) == 3
    assert df["is_duplicate"].sum() == 1
    dup_row = df[df["is_duplicate"]].iloc[0]
    assert dup_row["first_of_hash_id"] == "1"


def test_build_manifest_flags_unreadable_files(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    _write_jpg(tmp_path / "good.jpg", np.zeros((16, 16)))
    df = build_manifest(tmp_path)
    statuses = dict(zip(df["image_id"], df["read_status"]))
    assert statuses["good"] == "ok"
    assert statuses["bad"] != "ok"


def test_manifest_round_trips_through_parquet(tmp_path):
    _write_jpg(tmp_path / "a.jpg", np.zeros((16, 16)))
    df = build_manifest(tmp_path)
    out_path = tmp_path / "manifest.parquet"
    write_manifest(df, out_path)
    df2 = read_manifest(out_path)
    assert list(df2["image_id"]) == list(df["image_id"])
