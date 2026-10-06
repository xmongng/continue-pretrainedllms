import zipfile

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from llm_training.data.dataset import discover_data_files


def test_discovers_jsonl_files_inside_zip(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("part-1.jsonl", '{"text": "one"}\n')
        archive.writestr("nested/part-2.json", '[{"text": "two"}]')
        archive.writestr("notes.txt", "ignored")

    files = discover_data_files(archive_path, tmp_path / "cache")

    assert len(files) == 2
    assert {file.rsplit(".", 1)[-1] for file in files} == {"json", "jsonl"}


def test_rejects_zip_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.jsonl", '{"text": "unsafe"}\n')

    with pytest.raises(ValueError, match="Unsafe path"):
        discover_data_files(archive_path, tmp_path / "cache")
