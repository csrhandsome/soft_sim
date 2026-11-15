"""Capture native child stdout/stderr while retaining the actual exit code."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

from soft_sim.config import ROOT


class RunLogging(unittest.TestCase):
    def test_run_with_spaces_keeps_preamble_and_stderr_and_failure_status(self):
        spec = importlib.util.spec_from_file_location('soft_sim_dispatch', ROOT / 'scripts/run.py')
        dispatch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dispatch)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / 'run with spaces'
            run.mkdir()
            code = ("import sys\n"
                    "print('preamble', flush=True)\n"
                    f"print('RUN test ' + {str(run)!r}, flush=True)\n"
                    "print('native diagnostic', file=sys.stderr, flush=True)\n"
                    "raise SystemExit(3)\n")
            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                status = dispatch.run_with_log([sys.executable, '-c', code], None)
            self.assertEqual(status, 3)
            text = (run / 'logs/simulation.log').read_text()
            self.assertEqual(text, terminal.getvalue())
            self.assertIn('preamble', text)
            self.assertIn('native diagnostic', text)
