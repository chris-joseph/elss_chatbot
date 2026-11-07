"""
test_overnight_sync.py
Tests for overnight_update orchestrator and backup/rollback mechanism.
"""

import os
import shutil
import pytest
from scripts.overnight_update import create_backup, rollback, trigger_hot_reload, RELOAD_TRIGGER_FILE

def test_backup_and_rollback():
    """Tests creating an atomic snapshot backup and verifying rollback behavior."""
    backup_path = create_backup()
    assert os.path.exists(backup_path)
    assert os.path.isdir(os.path.join(backup_path, "sources"))
    
    # Verify rollback executes without error
    rollback(backup_path)

def test_trigger_hot_reload():
    """Verifies that trigger_hot_reload creates or updates the .reload_trigger file."""
    trigger_hot_reload()
    assert os.path.exists(RELOAD_TRIGGER_FILE)
    with open(RELOAD_TRIGGER_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 10
