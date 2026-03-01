import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.providers.openai_codex_provider import _strip_model_prefix
from nanobot.providers.registry import find_by_model

runner = CliRunner()


@pytest.fixture
def mock_paths():
    """Mock config/workspace paths for test isolation."""
    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.utils.helpers.get_workspace_path") as mock_ws:

        base_dir = Path("./test_onboard_data")
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_sc.side_effect = lambda config: config_file.write_text("{}")

        yield config_file, workspace_dir

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "memory" / "MEMORY.md").exists()


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()


def test_config_matches_github_copilot_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "github-copilot/gpt-5.3-codex"

    assert config.get_provider_name() == "github_copilot"


def test_config_matches_openai_codex_with_hyphen_prefix():
    config = Config()
    config.agents.defaults.model = "openai-codex/gpt-5.1-codex"

    assert config.get_provider_name() == "openai_codex"


def test_find_by_model_prefers_explicit_prefix_over_generic_codex_keyword():
    spec = find_by_model("github-copilot/gpt-5.3-codex")

    assert spec is not None
    assert spec.name == "github_copilot"


def test_litellm_provider_canonicalizes_github_copilot_hyphen_prefix():
    provider = LiteLLMProvider(default_model="github-copilot/gpt-5.3-codex")

    resolved = provider._resolve_model("github-copilot/gpt-5.3-codex")

    assert resolved == "github_copilot/gpt-5.3-codex"


def test_openai_codex_strip_prefix_supports_hyphen_and_underscore():
    assert _strip_model_prefix("openai-codex/gpt-5.1-codex") == "gpt-5.1-codex"
    assert _strip_model_prefix("openai_codex/gpt-5.1-codex") == "gpt-5.1-codex"


# ============================================================================
# provider set-key
# ============================================================================


@pytest.fixture
def config_paths(tmp_path):
    """Fixture that wires load_config/save_config/get_config_path to a temp dir."""
    config_file = tmp_path / "config.json"
    saved: list[Config] = []

    def _load(_path=None):
        return Config()

    def _save(cfg, _path=None):
        saved.append(cfg)
        config_file.write_text("{}")

    with patch("nanobot.config.loader.load_config", side_effect=_load), \
         patch("nanobot.config.loader.save_config", side_effect=_save), \
         patch("nanobot.config.loader.get_config_path", return_value=config_file):
        yield config_file, saved


def test_provider_set_key_kilo(config_paths):
    """set-key stores the API key for the kilo provider."""
    config_file, saved = config_paths

    result = runner.invoke(app, ["provider", "set-key", "kilo", "sk-kilo-test"])

    assert result.exit_code == 0, result.stdout
    assert "Kilo.ai" in result.stdout
    assert len(saved) == 1
    assert saved[0].providers.kilo.api_key == "sk-kilo-test"


def test_provider_set_key_with_api_base(config_paths):
    """set-key stores both API key and custom base URL."""
    config_file, saved = config_paths

    result = runner.invoke(
        app,
        ["provider", "set-key", "kilo", "sk-kilo-test", "--api-base", "https://custom.kilo.ai/v1"],
    )

    assert result.exit_code == 0, result.stdout
    assert saved[0].providers.kilo.api_key == "sk-kilo-test"
    assert saved[0].providers.kilo.api_base == "https://custom.kilo.ai/v1"


def test_provider_set_key_unknown_provider(config_paths):
    """set-key exits with error for unknown provider names."""
    result = runner.invoke(app, ["provider", "set-key", "nonexistent", "some-key"])

    assert result.exit_code != 0


def test_provider_set_key_oauth_provider_rejected(config_paths):
    """set-key rejects OAuth-only providers with a helpful message."""
    result = runner.invoke(app, ["provider", "set-key", "openai-codex", "some-key"])

    assert result.exit_code != 0
    assert "OAuth" in result.stdout or "login" in result.stdout


# ============================================================================
# provider use
# ============================================================================


def test_provider_use_kilo(config_paths):
    """use switches the active provider to kilo."""
    config_file, saved = config_paths

    result = runner.invoke(app, ["provider", "use", "kilo"])

    assert result.exit_code == 0, result.stdout
    assert "Kilo.ai" in result.stdout
    assert saved[0].agents.defaults.provider == "kilo"


def test_provider_use_with_model(config_paths):
    """use sets both provider and model when --model is supplied."""
    config_file, saved = config_paths

    result = runner.invoke(app, ["provider", "use", "kilo", "--model", "claude-opus-4-5"])

    assert result.exit_code == 0, result.stdout
    assert saved[0].agents.defaults.provider == "kilo"
    assert saved[0].agents.defaults.model == "claude-opus-4-5"


def test_provider_use_unknown_provider(config_paths):
    """use exits with error for unknown provider names."""
    result = runner.invoke(app, ["provider", "use", "nonexistent"])

    assert result.exit_code != 0
