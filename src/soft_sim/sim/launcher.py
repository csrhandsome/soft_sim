"""Local-window and official headless WebRTC launch modes for Isaac Sim."""

import os
from pathlib import Path
import sys


def add_livestream_arguments(parser):
    parser.add_argument('--livestream', action='store_true',
                        help='Stream the full Isaac Sim UI over WebRTC without DISPLAY')
    parser.add_argument('--livestream-ip',
                        help='Server IPv4 address reachable from the client (LAN/VPN/public IP)')
    parser.add_argument('--livestream-signal-port', type=int, help='WebRTC TCP signaling port (default: 49100)')
    parser.add_argument('--livestream-stream-port', type=int, help='WebRTC UDP media port (default: 47998)')


def configure_visualization(parser, args, simulation):
    simulation.livestream = args.livestream or simulation.livestream
    for name in ('livestream_ip', 'livestream_signal_port', 'livestream_stream_port'):
        value = getattr(args, name)
        if value is not None:
            setattr(simulation, name, value)
    if args.gui and simulation.livestream:
        parser.error('--gui and --livestream are separate visualization modes; select one')
    if not simulation.livestream and any(getattr(args, name) is not None for name in (
            'livestream_ip', 'livestream_signal_port', 'livestream_stream_port')):
        parser.error('--livestream-ip/ports require --livestream (or simulation.livestream in the config)')
    if simulation.livestream:
        simulation.headless = True
    elif args.gui and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        parser.error('--gui requires DISPLAY. For remote SSH visualization use --livestream --livestream-ip SERVER_IP')


def create_simulation_app(simulation):
    # Load one complete charset_normalizer before Kit adds its bundled copies.
    # Mixing their compiled cd/md modules causes ABI errors in core extensions.

    # Importing isaacsim initializes EXP_PATH; import Kit/Omni APIs only after app startup.
    from isaacsim import SimulationApp

    launch = {'headless': simulation.headless, 'renderer': 'RaytracedLighting'}
    if not simulation.livestream:
        return SimulationApp(launch)

    experience = Path(os.environ['EXP_PATH']) / 'isaacsim.exp.full.streaming.kit'
    if not experience.is_file():
        raise FileNotFoundError(f'Official Isaac Sim streaming experience is missing: {experience}')
    settings = '/exts/omni.kit.livestream.app/primaryStream'
    extra_args = [f'--{settings}/signalPort={simulation.livestream_signal_port}',
                  f'--{settings}/streamPort={simulation.livestream_stream_port}']
    if simulation.livestream_ip:
        extra_args.append(f'--{settings}/publicIp={simulation.livestream_ip}')
    # Headless must keep the UI visible in the encoded framebuffer.
    launch.update(headless=True, hide_ui=False, disable_viewport_updates=False, extra_args=extra_args)
    print(f'Starting official WebRTC experience: {experience.name}; '
          f'TCP {simulation.livestream_signal_port}, UDP {simulation.livestream_stream_port}', flush=True)
    app = SimulationApp(launch, experience=str(experience))
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled('omni.kit.livestream.app'):
        app.close()
        raise RuntimeError('Official WebRTC livestream extension did not start; check Isaac Sim logs')
    host = simulation.livestream_ip or '<server IP reachable from your client>'
    print(f'WebRTC experience initialized. Connect the Isaac Sim WebRTC Streaming Client to {host}. '
          'Check the stream-server log for the actual listening ports; app ready alone does not verify a connection.',
          flush=True)
    return app


def keep_open(app, render=None):
    print('Keeping the scene open for viewing. Press Ctrl+C in the SSH terminal to exit.', flush=True)
    try:
        while app.is_running():
            # World.render() refreshes Kit without advancing the labeled physics state.
            if render is None:
                app.update()
            else:
                render()
    except KeyboardInterrupt:
        pass
