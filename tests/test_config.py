import pytest

from metric_study.config import ConfigError, StudyConfig

CONFIG_PATH = "configs/metric_study.yaml"


def test_load_default_section():
    cfg = StudyConfig.load(CONFIG_PATH, "default")
    assert cfg.seed == 0
    assert "native" in cfg.resolutions


def test_load_dry_run_section_inherits_and_overrides():
    cfg = StudyConfig.load(CONFIG_PATH, "dry_run")
    assert cfg.chunk_size == 10
    assert cfg.seed == 0  # inherited from &default


def test_missing_section_raises():
    with pytest.raises(ConfigError):
        StudyConfig.load(CONFIG_PATH, "not_a_real_section")


def test_config_hash_is_stable_and_order_independent():
    cfg1 = StudyConfig.load(CONFIG_PATH, "default")
    cfg2 = StudyConfig.load(CONFIG_PATH, "default")
    assert cfg1.config_hash == cfg2.config_hash

    reordered = dict(reversed(list(cfg1.params.items())))
    cfg3 = StudyConfig(cfg1.yaml_path, "default", reordered)
    assert cfg3.config_hash == cfg1.config_hash


def test_config_hash_changes_with_content():
    cfg1 = StudyConfig.load(CONFIG_PATH, "default")
    modified = dict(cfg1.params)
    modified["seed"] = 999
    cfg2 = StudyConfig(cfg1.yaml_path, "default", modified)
    assert cfg1.config_hash != cfg2.config_hash


def test_validate_rejects_missing_key():
    cfg = StudyConfig.load(CONFIG_PATH, "default")
    bad = dict(cfg.params)
    del bad["seed"]
    bad_cfg = StudyConfig(cfg.yaml_path, "default", bad)
    with pytest.raises(ConfigError):
        bad_cfg.validate()
