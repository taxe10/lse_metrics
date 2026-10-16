import numpy as np
from PIL import Image

from metric_study.io import iter_image_paths, read_image


def test_read_image_success(tmp_path):
    path = tmp_path / "frame.png"
    arr = (np.random.default_rng(0).uniform(0, 255, size=(32, 32))).astype(np.uint8)
    Image.fromarray(arr).save(path)

    result = read_image(path)
    assert result.ok
    assert result.image.shape == (32, 32)
    assert result.meta["bit_depth"] == 8


def test_read_image_missing_file(tmp_path):
    result = read_image(tmp_path / "does_not_exist.png")
    assert not result.ok
    assert "missing_file" in result.status


def test_read_image_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not a real image")
    result = read_image(path)
    assert not result.ok


def test_iter_image_paths_recursive_and_filtered(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "sub" / "b.png").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_bytes(b"x")
    paths = sorted(p.name for p in iter_image_paths(tmp_path))
    assert paths == ["a.jpg", "b.png"]
