from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanobot.skill_loader import SkillsLoader


def _make_skill(directory: Path, name: str, content: str) -> Path:
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    return skill_file


class TestSkillsLoaderInit:
    def test_default_builtin_dir(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        assert loader.workspace == tmp_path
        assert loader.workspace_skills == tmp_path / "skills"

    def test_custom_builtin_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "builtin"
        loader = SkillsLoader(tmp_path, builtin_skills_dir=custom)
        assert loader.builtin_skills == custom


class TestListSkills:
    def test_empty_when_no_skills(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
        assert loader.list_skills() == []

    def test_lists_builtin_skills(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "weather", "# Weather Skill\n")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        skills = loader.list_skills(filter_unavailable=False)
        assert len(skills) == 1
        assert skills[0]["name"] == "weather"
        assert skills[0]["source"] == "builtin"

    def test_lists_workspace_skills(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "skills"
        _make_skill(ws_skills, "custom-skill", "# Custom\n")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")

        skills = loader.list_skills(filter_unavailable=False)
        assert len(skills) == 1
        assert skills[0]["name"] == "custom-skill"
        assert skills[0]["source"] == "workspace"

    def test_workspace_overrides_builtin(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "skills"
        builtin = tmp_path / "builtin"
        _make_skill(ws_skills, "weather", "# Workspace Weather\n")
        _make_skill(builtin, "weather", "# Builtin Weather\n")

        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)
        skills = loader.list_skills(filter_unavailable=False)
        assert len(skills) == 1
        assert skills[0]["source"] == "workspace"

    def test_combined_sources(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "skills"
        builtin = tmp_path / "builtin"
        _make_skill(ws_skills, "my-skill", "# My\n")
        _make_skill(builtin, "builtin-skill", "# Builtin\n")

        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)
        skills = loader.list_skills(filter_unavailable=False)
        names = {s["name"] for s in skills}
        assert names == {"my-skill", "builtin-skill"}

    def test_ignores_dirs_without_skill_md(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        (builtin / "empty_dir").mkdir(parents=True)
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)
        assert loader.list_skills(filter_unavailable=False) == []


class TestLoadSkill:
    def test_loads_workspace_skill(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "skills"
        _make_skill(ws_skills, "my-skill", "# My Skill Content\n")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")

        content = loader.load_skill("my-skill")
        assert content == "# My Skill Content\n"

    def test_loads_builtin_skill(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "weather", "# Weather Content\n")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        content = loader.load_skill("weather")
        assert content == "# Weather Content\n"

    def test_workspace_takes_priority(self, tmp_path: Path) -> None:
        ws_skills = tmp_path / "skills"
        builtin = tmp_path / "builtin"
        _make_skill(ws_skills, "skill", "# Workspace Version\n")
        _make_skill(builtin, "skill", "# Builtin Version\n")

        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)
        assert loader.load_skill("skill") == "# Workspace Version\n"

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
        assert loader.load_skill("nonexistent") is None


class TestLoadSkillsForContext:
    def test_loads_multiple_skills(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "weather", "# Weather\nSunny today.")
        _make_skill(builtin, "cron", "# Cron\nSchedule tasks.")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        result = loader.load_skills_for_context(["weather", "cron"])
        assert "### Skill: weather" in result
        assert "### Skill: cron" in result
        assert "Sunny today." in result

    def test_strips_frontmatter(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        content = "---\nname: test\ndescription: A test skill\n---\n# Real Content\n"
        _make_skill(builtin, "test-skill", content)
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        result = loader.load_skills_for_context(["test-skill"])
        assert "---" not in result
        assert "# Real Content" in result

    def test_empty_for_missing_skills(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
        assert loader.load_skills_for_context(["nonexistent"]) == ""


class TestBuildSkillsSummary:
    def test_produces_xml_format(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(
            builtin, "weather", "---\nname: weather\ndescription: Get weather\n---\n# Weather\n"
        )
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        summary = loader.build_skills_summary()
        assert "<skills>" in summary
        assert "</skills>" in summary
        assert "<name>weather</name>" in summary
        assert "<description>Get weather</description>" in summary

    def test_empty_when_no_skills(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
        assert loader.build_skills_summary() == ""

    def test_escapes_xml_special_chars(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "test", '---\ndescription: A <test> & "skill"\n---\n# Test\n')
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        summary = loader.build_skills_summary()
        assert "&lt;" in summary
        assert "&gt;" in summary
        assert "&amp;" in summary


class TestGetSkillMetadata:
    def test_parses_frontmatter(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        content = "---\nname: weather\ndescription: Get weather\n---\n# Weather\n"
        _make_skill(builtin, "weather", content)
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        meta = loader.get_skill_metadata("weather")
        assert meta is not None
        assert meta["name"] == "weather"
        assert meta["description"] == "Get weather"

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
        assert loader.get_skill_metadata("nonexistent") is None

    def test_returns_none_when_no_frontmatter(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "plain", "# Plain Skill\nNo frontmatter.")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        assert loader.get_skill_metadata("plain") is None


class TestRequirements:
    def test_skill_with_no_requirements_available(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "basic", "# Basic\n")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        skills = loader.list_skills(filter_unavailable=True)
        assert any(s["name"] == "basic" for s in skills)

    def test_skill_with_missing_bin_filtered(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        content = '---\nmetadata: {"nanobot":{"requires":{"bins":["nonexistent_tool_xyz"]}}}\n---\n# Needs tool\n'
        _make_skill(builtin, "needs-tool", content)
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        filtered = loader.list_skills(filter_unavailable=True)
        assert not any(s["name"] == "needs-tool" for s in filtered)

        unfiltered = loader.list_skills(filter_unavailable=False)
        assert any(s["name"] == "needs-tool" for s in unfiltered)

    def test_skill_with_missing_env_filtered(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        content = '---\nmetadata: {"nanobot":{"requires":{"env":["NONEXISTENT_ENV_VAR_XYZ"]}}}\n---\n# Needs env\n'
        _make_skill(builtin, "needs-env", content)
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        filtered = loader.list_skills(filter_unavailable=True)
        assert not any(s["name"] == "needs-env" for s in filtered)

    def test_skill_with_satisfied_env_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_TEST_SKILL_VAR", "1")
        builtin = tmp_path / "builtin"
        content = '---\nmetadata: {"nanobot":{"requires":{"env":["MY_TEST_SKILL_VAR"]}}}\n---\n# Has env\n'
        _make_skill(builtin, "has-env", content)
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        filtered = loader.list_skills(filter_unavailable=True)
        assert any(s["name"] == "has-env" for s in filtered)


class TestGetAlwaysSkills:
    def test_returns_always_skills(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "always-on", "---\nalways: true\n---\n# Always On\n")
        _make_skill(builtin, "on-demand", "# On Demand\n")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        always = loader.get_always_skills()
        assert "always-on" in always
        assert "on-demand" not in always

    def test_empty_when_none_marked(self, tmp_path: Path) -> None:
        builtin = tmp_path / "builtin"
        _make_skill(builtin, "regular", "# Regular\n")
        loader = SkillsLoader(tmp_path, builtin_skills_dir=builtin)

        assert loader.get_always_skills() == []


class TestStripFrontmatter:
    def test_strips_yaml_frontmatter(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        content = "---\nname: test\n---\n# Body\n"
        assert loader._strip_frontmatter(content) == "# Body"

    def test_no_frontmatter_returns_as_is(self, tmp_path: Path) -> None:
        loader = SkillsLoader(tmp_path)
        content = "# Just Content\n"
        assert loader._strip_frontmatter(content) == content
