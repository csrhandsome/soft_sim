"""Verify remote launch and render contracts without initializing Kit or a GPU."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from shared.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'backends' / 'isaac_sim' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_module('soft_sim_app_launcher', 'app_launcher.py')


class IsaacLivestreamContracts(unittest.TestCase):
    def test_streaming_launch_keeps_ui_visible_without_a_local_window(self):
        config = ExperimentConfig()
        config.simulation.livestream = True
        config.simulation.livestream_ip = '192.0.2.10'
        config.simulation.livestream_signal_port = 49200
        config.simulation.livestream_stream_port = 48100
        fake_isaac = ModuleType('isaacsim')
        fake_isaac.SimulationApp = Mock()
        fake_omni = ModuleType('omni')
        fake_omni.kit = ModuleType('omni.kit')
        fake_omni.kit.app = ModuleType('omni.kit.app')
        manager = Mock()
        manager.is_extension_enabled.return_value = True
        fake_omni.kit.app.get_app = Mock(return_value=SimpleNamespace(get_extension_manager=lambda: manager))
        modules = {'isaacsim': fake_isaac, 'omni': fake_omni, 'omni.kit': fake_omni.kit,
                   'omni.kit.app': fake_omni.kit.app}
        with tempfile.TemporaryDirectory() as directory:
            experience = Path(directory) / 'isaacsim.exp.full.streaming.kit'
            experience.touch()
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {'EXP_PATH': directory}), redirect_stdout(io.StringIO()):
                app = launcher.create_simulation_app(config.simulation)
            launch = fake_isaac.SimulationApp.call_args.args[0]
            self.assertEqual(fake_isaac.SimulationApp.call_args.kwargs['experience'], str(experience))
        self.assertIs(app, fake_isaac.SimulationApp.return_value)
        self.assertTrue(launch['headless'])
        self.assertFalse(launch['hide_ui'])
        self.assertFalse(launch['disable_viewport_updates'])
        self.assertIn('--/exts/omni.kit.livestream.app/primaryStream/publicIp=192.0.2.10', launch['extra_args'])
        self.assertIn('--/exts/omni.kit.livestream.app/primaryStream/signalPort=49200', launch['extra_args'])
        self.assertIn('--/exts/omni.kit.livestream.app/primaryStream/streamPort=48100', launch['extra_args'])

    def test_plain_headless_launch_does_not_enable_streaming(self):
        fake_isaac = ModuleType('isaacsim')
        fake_isaac.SimulationApp = Mock()
        with patch.dict(sys.modules, {'isaacsim': fake_isaac}):
            launcher.create_simulation_app(ExperimentConfig().simulation)
        launch = fake_isaac.SimulationApp.call_args.args[0]
        self.assertTrue(launch['headless'])
        self.assertNotIn('extra_args', launch)
        self.assertNotIn('experience', fake_isaac.SimulationApp.call_args.kwargs)

    def test_streaming_works_without_display_even_with_training_cameras_disabled(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gui', action='store_true')
        launcher.add_livestream_arguments(parser)
        args = parser.parse_args(['--livestream', '--livestream-ip', '192.0.2.10'])
        config = ExperimentConfig()
        config.simulation.render = False
        with patch.dict(os.environ, {}, clear=True):
            launcher.configure_visualization(parser, args, config.simulation)
        backend = load_module('soft_sim_isaac_backend', 'soft_gripper_env.py')
        env = object.__new__(backend.IsaacSimGripperEnv)
        env.config, env.app, env.world = config, Mock(), Mock()
        env.step_index = config.simulation.capture_every_steps
        env.step_physics()
        env.world.step.assert_called_once_with(render=False)
        env.world.render.assert_called_once_with()

    def test_viewing_after_collection_only_renders(self):
        app, render = Mock(), Mock()
        app.is_running.side_effect = [True, True, False]
        with redirect_stdout(io.StringIO()):
            launcher.keep_open(app, render=render)
        self.assertEqual(render.call_count, 2)
        app.update.assert_not_called()

    def test_collection_cli_passes_streaming_settings_and_keeps_scene_open(self):
        from shared import cli
        environment_class = Mock(backend_name='isaacsim')
        context = MagicMock()
        environment_class.return_value = context
        env = context.__enter__.return_value
        arguments = ['run.py', '--livestream', '--livestream-ip', '192.0.2.10', '--no-render']
        result = ('example/episode', {'frame_count': 1, 'peak_native_squeeze_force_n': None})
        with patch.object(sys, 'argv', arguments), patch.object(cli, 'collect_episode', return_value=result) as collect:
            with redirect_stdout(io.StringIO()):
                cli.main(environment_class, add_arguments=launcher.add_livestream_arguments,
                         configure_simulation=launcher.configure_visualization)
        simulation = environment_class.call_args.args[0].simulation
        self.assertTrue(simulation.livestream)
        self.assertTrue(simulation.headless)
        self.assertFalse(simulation.render)
        self.assertEqual(simulation.livestream_ip, '192.0.2.10')
        self.assertEqual(collect.call_args.args[0], env)
        env.wait_for_close.assert_called_once_with()
        context.__exit__.assert_called_once_with(None, None, None)

    def test_invalid_modes_fail_before_kit_startup(self):
        cases = [(['--gui'], 'requires DISPLAY'),
                 (['--gui', '--livestream'], 'select one'),
                 (['--livestream-ip', '192.0.2.10'], 'require --livestream')]
        environment = dict(os.environ)
        environment.pop('DISPLAY', None)
        for script in ('run.py', 'inspect_assets.py'):
            for args, message in cases:
                with self.subTest(script=script, args=args):
                    result = subprocess.run([sys.executable, str(ROOT / 'backends' / 'isaac_sim' / script), *args],
                                            env=environment, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn('Starting kit application', result.stdout)

    def test_invalid_ports_are_rejected(self):
        for field in ('livestream_signal_port', 'livestream_stream_port'):
            for value in (0, 65536, True, 49100.5):
                config = ExperimentConfig()
                setattr(config.simulation, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    config.validate()


if __name__ == '__main__':
    unittest.main()
