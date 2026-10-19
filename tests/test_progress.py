import json
import subprocess
from datetime import date

import pytest

from grading import progress


@pytest.fixture
def pfile(tmp_path):
    return tmp_path / ".progress.json"


def test_first_record_creates_the_file_and_counts_the_attempt(pfile):
    result = progress.record("02", "answer", False, path=pfile)
    assert result == progress.RecordResult(attempt=1, module_complete=False, note="")
    entry = json.loads(pfile.read_text())["modules"]["02"]
    assert entry == {"passed_checks": [], "attempts": 1, "completed_on": None}


def test_a_pass_completes_the_module_and_stamps_the_date(pfile):
    progress.record("02", "answer", False, path=pfile)
    result = progress.record("02", "answer", True, path=pfile, today=date(2026, 10, 16))
    assert result.attempt == 2 and result.module_complete
    assert json.loads(pfile.read_text())["modules"]["02"]["completed_on"] == "2026-10-16"


def test_passes_stick_after_a_later_not_yet(pfile):
    progress.record("02", "answer", True, path=pfile, today=date(2026, 10, 16))
    result = progress.record("02", "answer", False, path=pfile, today=date(2026, 10, 17))
    assert result.module_complete and result.attempt == 2
    assert json.loads(pfile.read_text())["modules"]["02"]["completed_on"] == "2026-10-16"
    assert progress.completed_modules(path=pfile) == {"02"}


def test_module_04_needs_both_the_code_test_and_the_answer(pfile):
    assert not progress.record("04", "answer", True, path=pfile).module_complete
    assert progress.record("04", "code", True, path=pfile).module_complete
    assert progress.completed_modules(path=pfile) == {"04"}


def test_an_unreadable_file_is_moved_aside_and_progress_starts_fresh(pfile):
    pfile.write_text("{not json")
    result = progress.record("01", "answer", True, path=pfile)
    assert result.note == progress.UNREADABLE_NOTE and result.attempt == 1
    assert pfile.with_name(".progress.json.bak").read_text() == "{not json"
    assert json.loads(pfile.read_text())["modules"]["01"]["attempts"] == 1


def test_valid_json_with_the_wrong_shape_is_treated_as_unreadable(pfile):
    pfile.write_text(json.dumps({"modules": ["01", "02"]}))
    result = progress.record("01", "answer", True, path=pfile)
    assert result.note == progress.UNREADABLE_NOTE
    assert pfile.with_name(".progress.json.bak").exists()


def test_an_entry_missing_completed_on_is_treated_as_unreadable(pfile):
    # A file missing the "completed_on" key (rather than having it set to
    # null) used to pass _valid() and then crash record() with a KeyError,
    # leaving the broken file in place so every later check() crashed too.
    pfile.write_text(json.dumps(
        {"version": 1, "modules": {"01": {"passed_checks": ["answer"], "attempts": 2}}}))
    result = progress.record("01", "answer", True, path=pfile)
    assert result.note == progress.UNREADABLE_NOTE
    assert pfile.with_name(".progress.json.bak").exists()


def test_an_unwritable_progress_file_never_breaks_grading(tmp_path):
    unwritable = tmp_path / "no-such-folder" / ".progress.json"
    result = progress.record("02", "answer", True, path=unwritable)
    assert result.module_complete and result.attempt == 1
    assert result.note == progress.UNWRITABLE_NOTE


def test_save_removes_the_temp_file_if_replacing_it_fails(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(progress.os, "replace", boom)
    pfile = tmp_path / ".progress.json"
    with pytest.raises(OSError):
        progress._save(pfile, {"version": 1, "modules": {}})
    assert list(tmp_path.iterdir()) == []


def test_record_reports_unwritable_and_leaves_no_tmp_file_when_replace_fails(pfile, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(progress.os, "replace", boom)
    result = progress.record("02", "answer", True, path=pfile)
    assert result.note == progress.UNWRITABLE_NOTE
    assert not pfile.with_name(pfile.name + ".tmp").exists()


def test_no_temporary_file_is_left_behind(pfile):
    progress.record("02", "answer", True, path=pfile)
    assert sorted(p.name for p in pfile.parent.iterdir()) == [".progress.json"]


def test_reset_removes_the_file(pfile):
    progress.record("02", "answer", True, path=pfile)
    progress.reset(path=pfile)
    assert not pfile.exists()
    assert progress.completed_modules(path=pfile) == set()


def test_reset_without_a_file_is_fine(pfile):
    progress.reset(path=pfile)


def test_default_path_is_read_at_call_time(monkeypatch, pfile):
    monkeypatch.setattr(progress, "PROGRESS_FILE", pfile)
    progress.record("03", "answer", True)
    assert progress.completed_modules() == {"03"}
    progress.reset()
    assert not pfile.exists()


def test_module_info_reads_the_number_and_title_from_the_readme(tmp_path):
    folder = tmp_path / "02-prompt-engineering"
    folder.mkdir()
    (folder / "README.md").write_text("# Module 02: Prompt Engineering\n\nText\n")
    assert progress.module_info(folder) == ("02", "Prompt Engineering")


def test_module_info_falls_back_to_the_folder_name(tmp_path):
    folder = tmp_path / "05-github-basics"
    folder.mkdir()
    assert progress.module_info(folder) == ("05", "Github Basics")


def test_module_info_tolerates_undecodable_bytes_after_the_first_line(tmp_path):
    # On Windows (cp1252 default) a README with UTF-8 bytes like \x9d used to
    # raise UnicodeDecodeError, which escaped check() because only
    # (OSError, IndexError) was caught.
    folder = tmp_path / "02-prompt-engineering"
    folder.mkdir()
    (folder / "README.md").write_bytes(b"# Module 02: Prompt Engineering\n\xff\xfe\x9d")
    assert progress.module_info(folder) == ("02", "Prompt Engineering")


def test_module_info_falls_back_to_the_slug_when_the_first_line_itself_is_undecodable(tmp_path):
    folder = tmp_path / "02-github-basics"
    folder.mkdir()
    (folder / "README.md").write_bytes(b"# Module 0\xff2: Something\nText\n")
    assert progress.module_info(folder) == ("02", "Github Basics")


def test_module_info_outside_a_module_folder(tmp_path):
    assert progress.module_info(tmp_path) == (None, "Exercise")
    assert progress.module_info(tmp_path / "_template") == (None, "Exercise")


def test_list_modules_finds_numbered_folders_only(tmp_path):
    for name in ("02-b", "01-a", "_template", "notes"):
        (tmp_path / name).mkdir()
    (tmp_path / "03-c.md").write_text("")
    assert progress.list_modules(tmp_path) == ["01", "02"]


def test_list_modules_of_a_missing_folder_is_empty(tmp_path):
    assert progress.list_modules(tmp_path / "nope") == []


def test_list_modules_of_the_real_repo():
    assert progress.list_modules()[:4] == ["01", "02", "03", "04"]


def test_progress_files_are_git_ignored():
    for name in (".progress.json", ".progress.json.bak", ".progress.json.tmp"):
        assert subprocess.run(["git", "check-ignore", "-q", name], cwd=progress.REPO_ROOT).returncode == 0
