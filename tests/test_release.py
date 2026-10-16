from tests.support import config_path
"""Release gates: installed entrypoints, diagnostics and one full hybrid smoke run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from engine import __version__, MANIFEST_SCHEMA_VERSION
from engine.galerkin.run import SegmentedRun
from engine.model.loading import load_model
from engine.visualization.spectral import ArchiveSequence
from engine.archive import read_archive, validate, json_array

ROOT = Path(__file__).resolve().parents[1]


def small_config():
    c = json.loads((config_path('spectral_single.json')).read_text())
    c['grid']['shape'] = [24]
    c['solver']['modes'] = [4]
    c['t_end'] = .12
    c['split']['enabled'] = False
    return c


class ReleaseTests(unittest.TestCase):
    def test_golden_pipeline(self):
        """Config → fixed ODE → archive → birth → new ODE → reconstruction."""
        c = small_config()
        c['events'] = [{'time': .06, 'type': 'p_birth', 'birth_amplitude': 10}]
        with tempfile.TemporaryDirectory() as tmp:
            m = SegmentedRun(c, tmp).run()
            self.assertEqual(m['software_version'], __version__)
            self.assertEqual(m['run_manifest_schema_version'], MANIFEST_SCHEMA_VERSION)
            self.assertEqual(len(m['segments']), 2)
            self.assertEqual(m['segments'][0]['closure_reasons'], ['p_birth'])
            a, b = [read_archive(Path(tmp)/s['archive']) for s in m['segments']]
            self.assertEqual(b['coefficients'].shape[1], 2*a['coefficients'].shape[1])
            seq = ArchiveSequence(Path(tmp)/'manifest.json')
            values, _, _, _ = seq.frame(len(a['time']))
            np.testing.assert_allclose(values['P_0'], 10, atol=1e-12)
            self.assertEqual(m['relations'][0]['to'], 1)
            self.assertTrue(all(not x.dtype.hasobject for x in b.values()))

    def test_main_cli_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp/'input.json'; cfg.write_text(json.dumps(small_config()))
            env = os.environ.copy(); env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, '-m', 'engine', 'run',
                '--config', str(cfg), '--output', str(tmp/'run')],
                cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('segment=000', result.stderr)
            self.assertTrue((tmp/'run/manifest.json').is_file())
            self.assertEqual({x.name for x in tmp.iterdir()}, {'input.json', 'run'})

    def test_output_required(self):
        result = subprocess.run([sys.executable, '-m', 'engine', 'run', '--config',
            str(config_path('spectral_single.json'))], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--output', result.stderr)

    def test_unknown_model_message(self):
        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            load_model({'model': 'missing_release_fixture'})

    def test_broken_manifest_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.json'
            for payload in ({}, {'segments': [{}], 'field_registry': []},
                    {'segments': [{'id': 0, 'archive': 'missing.npz'}], 'field_registry': []}):
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, '[Mm]anifest'):
                    ArchiveSequence(path)

    def test_invalid_archive_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = SegmentedRun(small_config(), tmp).run()
            data = read_archive(Path(tmp)/m['segments'][0]['archive'])
            data['metadata_json'] = json_array([])
            with self.assertRaisesRegex(ValueError, 'metadata_json'):
                validate(data)
            data['metadata_json'] = json_array({})
            data['basis_json'] = json_array({})
            with self.assertRaisesRegex(ValueError, 'basis_json'):
                validate(data)
