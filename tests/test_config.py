"""
Configuration verification tests for Godrej Warehouse AI.
Validates that Streamlit configuration files and upload size thresholds are properly configured.
"""

from pathlib import Path
import streamlit.config as config


def test_streamlit_config_file_exists():
    """Verifies that .streamlit/config.toml exists at the project root."""
    project_root = Path(__file__).resolve().parent.parent
    config_file = project_root / ".streamlit" / "config.toml"
    assert config_file.exists(), ".streamlit/config.toml must exist in project root"
    assert config_file.is_file(), ".streamlit/config.toml must be a file"


def test_streamlit_server_upload_limit_configured():
    """Verifies that Streamlit server maxUploadSize and maxMessageSize are configured to 2048 MB."""
    max_upload_size = config.get_option("server.maxUploadSize")
    max_message_size = config.get_option("server.maxMessageSize")

    assert max_upload_size == 2048, f"server.maxUploadSize expected 2048, got {max_upload_size}"
    assert max_message_size == 2048, f"server.maxMessageSize expected 2048, got {max_message_size}"
