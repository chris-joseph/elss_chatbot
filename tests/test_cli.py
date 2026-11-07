"""
Unit tests for CLI runner and evaluator.
"""

from unittest.mock import patch, MagicMock
import pytest
from cli import run_eval, main
from rag.engine import RAGEngine

def test_run_eval_execution(capsys):
    mock_engine = MagicMock(spec=RAGEngine)
    mock_engine.answer.return_value = {
        "answer": "Test answer ([Source](https://groww.in)) Last updated from sources: 2026-11-07",
        "status": "factual",
        "latency_ms": 12.5,
        "source_url": "https://groww.in"
    }
    
    run_eval(mock_engine)
    captured = capsys.readouterr()
    assert "HEADLESS EVALUATION" in captured.out
    assert "Test answer" in captured.out
    assert "Source Link: https://groww.in" in captured.out
    assert mock_engine.answer.call_count == 10

def test_cli_main_with_eval_flag():
    with patch("sys.argv", ["cli.py", "--eval"]), patch("cli.run_eval") as mock_eval:
        main()
        mock_eval.assert_called_once()

def test_cli_main_interactive_mode():
    with patch("sys.argv", ["cli.py"]), patch("cli.run_interactive") as mock_interactive:
        main()
        mock_interactive.assert_called_once()

def test_run_interactive_loop(capsys):
    from cli import run_interactive
    mock_engine = MagicMock(spec=RAGEngine)
    mock_engine.answer.return_value = {
        "answer": "Expense ratio is 1.47%.",
        "status": "factual",
        "latency_ms": 10.0
    }
    
    with patch("builtins.input", side_effect=["What is expense ratio?", "quit"]):
        run_interactive(mock_engine)
        captured = capsys.readouterr()
        assert "CLI Mode" in captured.out
        assert "Expense ratio is 1.47%." in captured.out
        assert "Goodbye!" in captured.out

def test_run_interactive_keyboard_interrupt(capsys):
    from cli import run_interactive
    mock_engine = MagicMock(spec=RAGEngine)
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        run_interactive(mock_engine)
        captured = capsys.readouterr()
        assert "Exiting." in captured.out

def test_run_single_query(capsys):
    from cli import run_single_query
    mock_engine = MagicMock(spec=RAGEngine)
    mock_engine.answer.return_value = {
        "answer": "Test single query answer",
        "status": "factual",
        "source_url": "https://groww.in"
    }
    run_single_query(mock_engine, "sample question")
    captured = capsys.readouterr()
    assert "Test single query answer" in captured.out
    assert "Source: https://groww.in" in captured.out

def test_cli_main_with_query_flag():
    with patch("sys.argv", ["cli.py", "--query", "What is ELSS lock-in?"]), patch("cli.run_single_query") as mock_sq:
        main()
        mock_sq.assert_called_once()

def test_cli_main_with_provider_offline_eval():
    with patch("sys.argv", ["cli.py", "--eval", "--provider", "offline"]), patch("cli.run_eval") as mock_eval, patch("cli.RAGEngine") as mock_engine_cls:
        main()
        mock_engine_cls.assert_called_once_with(provider="offline", api_key=None, model=None)
        mock_eval.assert_called_once()

def test_cli_main_with_provider_offline_query():
    with patch("sys.argv", ["cli.py", "--query", "what is a mutual fund", "--provider", "offline"]), patch("cli.run_single_query") as mock_sq, patch("cli.RAGEngine") as mock_engine_cls:
        main()
        mock_engine_cls.assert_called_once_with(provider="offline", api_key=None, model=None)
        mock_sq.assert_called_once()


