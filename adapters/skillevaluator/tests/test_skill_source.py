"""Tests for locating the staged skill and keeping it inside ``/test_data``."""

import os

import pytest

import _skill_source as src


def _skill(path, name="s"):
    """Create a minimal skill directory containing ``SKILL.md``."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    return path


def test_not_mounted_fails_clearly(tmp_path):
    """A missing ``/test_data`` raises an error that says it is not mounted."""
    with pytest.raises(src.SkillSourceError, match="not mounted"):
        src.resolve_skill_path({}, test_data_dir=str(tmp_path / "missing"))


def test_root_is_the_skill(tmp_path):
    """When ``/test_data`` itself holds ``SKILL.md`` it is the skill."""
    _skill(tmp_path)
    assert src.resolve_skill_path({}, test_data_dir=str(tmp_path)) == tmp_path.resolve()


def test_single_nested_skill_is_discovered(tmp_path):
    """A single nested skill is found without ``skill_path``."""
    skill = _skill(tmp_path / "skills" / "alpha")
    assert src.resolve_skill_path({}, test_data_dir=str(tmp_path)) == skill.resolve()


def test_several_skills_are_ambiguous_and_listed(tmp_path):
    """Several skills raise an ambiguity error that lists them."""
    _skill(tmp_path / "a")
    _skill(tmp_path / "b")
    with pytest.raises(src.AmbiguousSkillError, match="2 skills found.*a, b"):
        src.resolve_skill_path({}, test_data_dir=str(tmp_path))


def test_no_skill_found(tmp_path):
    """A tree with no skill raises an error."""
    (tmp_path / "docs").mkdir()
    with pytest.raises(src.SkillSourceError, match="no skill"):
        src.resolve_skill_path({}, test_data_dir=str(tmp_path))


def test_explicit_relative_path_selects_one_of_many(tmp_path):
    """``skill_path`` selects one skill among several."""
    _skill(tmp_path / "a")
    b = _skill(tmp_path / "b")
    assert (
        src.resolve_skill_path({"skill_path": "b"}, test_data_dir=str(tmp_path))
        == b.resolve()
    )


def test_explicit_path_is_authoritative_and_never_falls_back(tmp_path):
    """An explicit ``skill_path`` is never replaced by discovery."""
    _skill(tmp_path / "a")
    with pytest.raises(src.SkillSourceError, match="not a directory"):
        src.resolve_skill_path({"skill_path": "nope"}, test_data_dir=str(tmp_path))


def test_explicit_path_without_manifest_fails(tmp_path):
    """An explicit path without ``SKILL.md`` is rejected."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(src.SkillSourceError, match="no SKILL.md"):
        src.resolve_skill_path({"skill_path": "empty"}, test_data_dir=str(tmp_path))


@pytest.mark.parametrize("bad", ["../outside", "/etc", "a/../../outside"])
def test_path_traversal_is_rejected(tmp_path, bad):
    """``..`` and absolute paths outside ``/test_data`` are rejected."""
    root = tmp_path / "td"
    _skill(root / "a")
    _skill(tmp_path / "outside")
    with pytest.raises(src.SkillSourceError, match="escapes"):
        src.resolve_skill_path({"skill_path": bad}, test_data_dir=str(root))


@pytest.mark.parametrize("bad", ["", "   ", 7, ["a"]])
def test_skill_path_must_be_nonempty_string(tmp_path, bad):
    """``skill_path`` must be a non-empty string."""
    _skill(tmp_path / "a")
    with pytest.raises(src.SkillSourceError, match="non-empty string"):
        src.resolve_skill_path({"skill_path": bad}, test_data_dir=str(tmp_path))


def test_symlinked_skill_dir_pointing_outside_is_rejected(tmp_path):
    """A skill directory symlinked outside ``/test_data`` is rejected."""
    root = tmp_path / "td"
    root.mkdir()
    outside = _skill(tmp_path / "outside")
    os.symlink(outside, root / "link")
    with pytest.raises(src.SkillSourceError, match="escapes"):
        src.resolve_skill_path({"skill_path": "link"}, test_data_dir=str(root))


def test_symlinked_manifest_pointing_outside_is_rejected(tmp_path):
    """A ``SKILL.md`` symlinked outside ``/test_data`` is rejected."""
    root = tmp_path / "td"
    (root / "s").mkdir(parents=True)
    secret = tmp_path / "secret.md"
    secret.write_text("outside")
    os.symlink(secret, root / "s" / "SKILL.md")
    with pytest.raises(src.SkillSourceError, match="resolves outside"):
        src.resolve_skill_path({"skill_path": "s"}, test_data_dir=str(root))


def test_symlinked_file_inside_skill_pointing_outside_is_rejected(tmp_path):
    """A file symlinked out of ``/test_data`` from inside the skill is rejected."""
    root = tmp_path / "td"
    skill = _skill(root / "s")
    (skill / "scripts").mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    os.symlink(secret, skill / "scripts" / "token.txt")
    with pytest.raises(src.SkillSourceError, match=r"symlink.*scripts/token\.txt"):
        src.resolve_skill_path({"skill_path": "s"}, test_data_dir=str(root))


def test_symlinked_dir_inside_skill_pointing_outside_is_rejected(tmp_path):
    """A directory symlinked out of ``/test_data`` from inside the skill is rejected."""
    root = tmp_path / "td"
    skill = _skill(root / "s")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, skill / "data")
    with pytest.raises(src.SkillSourceError, match="symlink"):
        src.resolve_skill_path({"skill_path": "s"}, test_data_dir=str(root))


def test_symlink_inside_skill_staying_within_root_is_allowed(tmp_path):
    """A symlink that resolves inside ``/test_data`` is harmless and accepted."""
    root = tmp_path / "td"
    skill = _skill(root / "s")
    shared = root / "shared.txt"
    shared.write_text("inside")
    os.symlink(shared, skill / "shared.txt")
    assert (
        src.resolve_skill_path({"skill_path": "s"}, test_data_dir=str(root))
        == skill.resolve()
    )


def test_discovery_skips_skill_with_escaping_symlink(tmp_path):
    """Auto-discovery ignores a skill whose tree links outside ``/test_data``."""
    root = tmp_path / "td"
    good = _skill(root / "good")
    bad = _skill(root / "bad")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    os.symlink(secret, bad / "leak.txt")
    assert src.resolve_skill_path({}, test_data_dir=str(root)) == good.resolve()


def test_discovery_ignores_hidden_dirs_and_symlinked_dirs(tmp_path):
    """Discovery skips hidden and symlinked directories."""
    root = tmp_path / "td"
    real = _skill(root / "real")
    _skill(root / ".git" / "hooks-skill")
    outside = _skill(tmp_path / "outside")
    os.symlink(outside, root / "linked")
    assert src.resolve_skill_path({}, test_data_dir=str(root)) == real.resolve()


def test_discovery_depth_is_bounded(tmp_path):
    """Discovery does not descend past its depth limit."""
    deep = tmp_path.joinpath(*["d"] * (src.MAX_DISCOVERY_DEPTH + 2))
    _skill(deep)
    with pytest.raises(src.SkillSourceError, match="no skill"):
        src.resolve_skill_path({}, test_data_dir=str(tmp_path))
