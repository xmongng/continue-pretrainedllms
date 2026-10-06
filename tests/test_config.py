import json

import pytest

from llm_training.config import load_config


def make_config(tmp_path, precision="auto"):
    project = tmp_path / "project"
    configs = project / "configs"
    data = project / "data"
    configs.mkdir(parents=True)
    data.mkdir()
    (data / "corpus.jsonl").write_text('{"text": "hello"}\n', encoding="utf-8")
    config = {
        "model": {"name_or_path": "example/model"},
        "data": {
            "source": "data/corpus.jsonl",
            "max_length": 128,
            "validation_fraction": 0.1,
        },
        "training": {"output_dir": "outputs/run"},
        "runtime": {"precision": precision},
    }
    path = configs / "test.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, project


def test_load_config_resolves_project_paths(tmp_path):
    path, project = make_config(tmp_path)
    config = load_config(path)

    assert config["data"]["source"] == str((project / "data/corpus.jsonl").resolve())
    assert config["training"]["output_dir"] == str((project / "outputs/run").resolve())
    assert config["data"]["text_column"] == "text"


def test_load_config_rejects_unknown_precision(tmp_path):
    path, _ = make_config(tmp_path, precision="int4")

    with pytest.raises(ValueError, match="runtime.precision"):
        load_config(path)


def test_config_inheritance_merges_before_validation(tmp_path):
    path, project = make_config(tmp_path)
    base = path.with_name("base.json")
    child = path.with_name("child.json")
    source = json.loads(path.read_text())
    output = source["training"].pop("output_dir")
    base.write_text(json.dumps(source), encoding="utf-8")
    child.write_text(
        json.dumps({"extends": "base.json", "training": {"output_dir": output}}),
        encoding="utf-8",
    )

    config = load_config(child)

    assert config["training"]["output_dir"] == str((project / output).resolve())
    assert config["runtime"]["precision"] == "auto"
