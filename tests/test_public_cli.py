"""Public entrypoint and install-resource regressions."""
import importlib
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import unittest

import engine
from engine.cli import COMMANDS


class PublicCliTests(unittest.TestCase):
    def command(self, *args, cwd):
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env['MPLBACKEND'] = 'Agg'
        return subprocess.run([sys.executable, '-m', 'engine', *args], cwd=cwd,
                              env=env, capture_output=True, text=True)

    def test_all_command_help_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for command in [*COMMANDS, 'examples']:
                with self.subTest(command=command):
                    result = self.command(command, '--help', cwd=directory)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)
                    self.assertNotIn('RuntimeWarning', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bundled_examples_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            listing = self.command('examples', cwd=directory)
            self.assertEqual(listing.returncode, 0, listing.stderr)
            self.assertIn('provenance_demo_1d.json', listing.stdout)
            result = self.command('examples', 'provenance_demo_1d', '--output', 'demo.json', cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = (Path(directory)/'demo.json').read_bytes()
            again = self.command('examples', 'basic_3d', '--output', 'demo.json', cwd=directory)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((Path(directory)/'demo.json').read_bytes(), before)
            self.assertNotEqual(self.command('examples', '../README', '--output', 'bad', cwd=directory).returncode, 0)

    def test_imports_are_package_local_and_have_no_cli_side_effects(self):
        for module in pkgutil.walk_packages(engine.__path__, engine.__name__ + '.'):
            if module.name == 'engine.__main__':
                continue
            with self.subTest(module=module.name):
                importlib.import_module(module.name)

    def test_version_and_unknown_command(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.command('--version', cwd=directory)
            self.assertEqual(result.returncode, 0)
            self.assertIn(engine.__version__, result.stdout)
            self.assertNotEqual(self.command('unknown-command', cwd=directory).returncode, 0)
