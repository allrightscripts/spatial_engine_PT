"""Saved-time selection must not interpolate or cross event-side semantics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from engine.visualization.spectral import ArchiveSequence,plot_spectral
from engine.galerkin.run import SegmentedRun
from tests.support import config_path


class TimeSelectionTests(unittest.TestCase):
    def sequence(self):
        seq=ArchiveSequence.__new__(ArchiveSequence)
        seq.frames=[(Path('a'),i,t,s) for i,(t,s) in enumerate([(0.,0),(.25,0),(.5,0),(.5,1),(.75,1),(1.,'terminal')])]
        return seq

    def test_nearest_and_event_sides(self):
        s=self.sequence()
        self.assertEqual(s.select_time(.5),3)
        self.assertEqual(s.select_time(.5,event_side='before'),2)
        self.assertEqual(s.select_time(.375),1)  # Equal distances: earlier time.
        self.assertEqual(s.select_time(.49),3)   # Nearest event, post-state by default.
        self.assertEqual(s.select_time(1.),5)
        self.assertEqual(s.select_time(0.),0)

    def test_directional_selection(self):
        s=self.sequence()
        self.assertEqual(s.select_time(.4,'before'),1)
        self.assertEqual(s.select_time(.4,'after'),3)
        self.assertEqual(s.select_time(.5,'before'),3) # time_mode != event_side.
        self.assertEqual(s.select_time(.5,'after','before'),2)

    def test_invalid_requests(self):
        s=self.sequence()
        for value in [-.1,1.1,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):s.select_time(value)
        with self.assertRaises(ValueError):s.select_time(.5,'interpolate')
        with self.assertRaises(ValueError):s.select_time(.5,event_side='middle')

    def test_single_frame(self):
        s=self.sequence();s.frames=s.frames[:1]
        self.assertEqual(s.select_time(0),0)
        with self.assertRaises(ValueError):s.select_time(.01)

    def test_real_archive_cli_and_api(self):
        c=json.loads(config_path('spectral_single.json').read_text())
        c.update(t_end=.1,save_dt=.025);c['grid']['shape']=[24];c['solver']['modes']=[4]
        c['split']['enabled']=False
        c['events']=[{'time':.05,'type':'p_birth','birth_amplitude':.2}]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);SegmentedRun(c,root/'run').run()
            manifest=root/'run/manifest.json';s=ArchiveSequence(manifest)
            before=s.select_time(.05,event_side='before');after=s.select_time(.05)
            self.assertEqual(len(s.frame(before)[0]),1);self.assertEqual(len(s.frame(after)[0]),2)
            result=plot_spectral(manifest,time=.051,save=root/'nested/frame.png')
            self.assertAlmostEqual(result['actual_time'],.05)
            self.assertEqual(result['frame_index'],after)
            self.assertGreater((root/'nested/frame.png').stat().st_size,100)
            with self.assertRaises(ValueError):plot_spectral(manifest,time=.05,frame_index=0)
            result=subprocess.run([sys.executable,'-m','engine','plot','--manifest',str(manifest),
                '--time','.051','--save',str(root/'cli.png')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('actual_time=',result.stdout);self.assertTrue((root/'cli.png').is_file())
            for options in [('--time','.05','--frame','0'),('--time-mode','before'),('--time','10')]:
                result=subprocess.run([sys.executable,'-m','engine','plot','--manifest',str(manifest),*options],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertNotIn('Traceback',result.stderr)
