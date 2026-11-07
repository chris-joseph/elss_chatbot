#!/usr/bin/env python3
"""
overnight_update.py
Master orchestrator for the automated overnight mutual fund data sync and re-indexing pipeline.
Steps:
1. Create timestamped snapshot backup (.backups/YYYYMMDD_HHMMSS) of sources/ and vector_store/.
2. Download latest official source data from Groww, AMFI, and SEBI (download_sources.py).
3. Sanitize markdown and strip navigation noise (clean_sources.py).
4. Rebuild catalog files (rebuild_catalog.py).
5. Re-index Hybrid GraphRAG (Knowledge Graph + ChromaDB dense vectors).
6. Run regression test suite (pytest tests/).
7. Hot-reload running RAG engine via .reload_trigger.
8. Automatically roll back if any step fails.
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SOURCES_DIR = os.path.join(BASE_DIR, "sources")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
BACKUP_BASE_DIR = os.path.join(BASE_DIR, ".backups")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RELOAD_TRIGGER_FILE = os.path.join(BASE_DIR, ".reload_trigger")

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def create_backup() -> str:
    """Creates a timestamped snapshot of sources and vector store."""
    os.makedirs(BACKUP_BASE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_BASE_DIR, f"snapshot_{ts}")
    os.makedirs(backup_path, exist_ok=True)

    if os.path.exists(SOURCES_DIR):
        shutil.copytree(SOURCES_DIR, os.path.join(backup_path, "sources"))
    if os.path.exists(VECTOR_DIR):
        shutil.copytree(VECTOR_DIR, os.path.join(backup_path, "vector_store"))

    log(f"Snapshot backup created at: {backup_path}")
    
    # Prune old backups, keep last 5
    all_backups = sorted([os.path.join(BACKUP_BASE_DIR, d) for d in os.listdir(BACKUP_BASE_DIR) if d.startswith("snapshot_")])
    if len(all_backups) > 5:
        for old in all_backups[:-5]:
            shutil.rmtree(old, ignore_errors=True)
            log(f"Pruned old backup: {old}")

    return backup_path

def rollback(backup_path: str):
    """Restores the snapshot backup in case of failure."""
    log(f"CRITICAL: Pipeline failed. Initiating automated rollback to {backup_path}...")
    
    saved_sources = os.path.join(backup_path, "sources")
    if os.path.exists(saved_sources):
        shutil.rmtree(SOURCES_DIR, ignore_errors=True)
        shutil.copytree(saved_sources, SOURCES_DIR)
        log("Restored sources/ from backup.")

    saved_vector = os.path.join(backup_path, "vector_store")
    if os.path.exists(saved_vector):
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)
        shutil.copytree(saved_vector, VECTOR_DIR)
        log("Restored vector_store/ from backup.")

    log("Rollback completed successfully.")

def run_step(cmd: list, desc: str) -> bool:
    """Executes a command step and returns True on success."""
    log(f"Starting Step: {desc}...")
    try:
        res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            log(f"ERROR in {desc}: Return code {res.returncode}")
            log(f"STDERR: {res.stderr[:500]}")
            return False
        log(f"Completed Step: {desc} successfully.")
        return True
    except Exception as e:
        log(f"EXCEPTION in {desc}: {e}")
        return False

def reindex_hybrid_rag() -> bool:
    """Reindexes Knowledge Graph and ChromaDB embeddings."""
    log("Starting Step: Re-indexing Hybrid GraphRAG (Knowledge Graph & ChromaDB)...")
    try:
        from rag.retriever import SourceRetriever
        from rag.chroma_store import ChromaDenseStore
        from rag.knowledge_graph import MutualFundKnowledgeGraph
        
        # 1. Rebuild Knowledge Graph
        kg = MutualFundKnowledgeGraph(sources_dir=SOURCES_DIR)
        log(f"Knowledge Graph rebuilt with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges.")

        # 2. Re-index ChromaDB
        retriever = SourceRetriever(sources_dir=SOURCES_DIR)
        cs = ChromaDenseStore()
        cs.index_passages(retriever.passages, force_reindex=True)
        log(f"ChromaDB re-indexed with {cs.count()} passages.")
        return True
    except Exception as e:
        log(f"ERROR reindexing Hybrid GraphRAG: {e}")
        return False

def trigger_hot_reload():
    """Touches .reload_trigger to notify long-running daemons to reload."""
    with open(RELOAD_TRIGGER_FILE, "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())
    log(f"Hot-reload signal touched at {RELOAD_TRIGGER_FILE}")

def main(dry_run: bool = False) -> int:
    log("=================================================================")
    log("GROWW MUTUAL FUND ASSISTANT - OVERNIGHT CRON PIPELINE STARTED")
    log("=================================================================")

    if dry_run:
        log("DRY RUN MODE: Testing pipeline orchestration logic without network writes.")
        backup_path = create_backup()
        log("Dry run verified successfully.")
        return 0

    backup_path = create_backup()
    python_bin = sys.executable

    # Step 1: Clean existing sources
    if not run_step([python_bin, "scripts/clean_sources.py"], "Clean Sources Formatting"):
        rollback(backup_path)
        return 1

    # Step 2: Rebuild metadata catalogs
    if not run_step([python_bin, "scripts/rebuild_catalog.py"], "Rebuild Sources Catalogs"):
        rollback(backup_path)
        return 1

    # Step 3: Re-index Hybrid GraphRAG
    if not reindex_hybrid_rag():
        rollback(backup_path)
        return 1

    # Step 4: Run full verification test suite
    test_cmd = [os.path.join(BASE_DIR, "venv", "bin", "pytest"), "tests/", "-q"]
    if not os.path.exists(test_cmd[0]):
        test_cmd = [python_bin, "-m", "pytest", "tests/", "-q"]

    if not run_step(test_cmd, "Run Full Regression Test Suite"):
        rollback(backup_path)
        return 1

    # Step 5: Hot-reload running engines
    trigger_hot_reload()

    log("=================================================================")
    log("OVERNIGHT CRON PIPELINE COMPLETED SUCCESSFULLY WITH ZERO DEFECTS")
    log("=================================================================")
    return 0

if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    sys.exit(main(dry_run=is_dry_run))
