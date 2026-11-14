"""Command table: which interpreter runs which module.

Standard library only, so ``scripts/run.py`` can read it before choosing an
interpreter. This is the single place that maps a command to an environment.
"""

COMMANDS = {
    "collect": dict(env="robodojo", module="soft_sim.cli.collect",
                    help="Plan, execute and stage a task episode in the original RoboDojo scene"),
    "export-scene": dict(env="robodojo", module="soft_sim.cli.collect",
                         help="Export the current official task scene without executing"),
    "replay": dict(env="robodojo", module="soft_sim.cli.collect",
                   help="Execute a saved plan in the matching scene"),
    "assembly": dict(env="robodojo", module="soft_sim.cli.collect",
                     help="Native-drive full-travel installation diagnostic"),
    "plan": dict(env="planning", module="soft_sim.cli.plan",
                 help="Plan a grasp on an exported scene with mplib"),
    "check-installation": dict(env="planning", module="soft_sim.cli.check_installation",
                               help="Offline installation geometry, gap and collision audit"),
    "prepare-mesh": dict(env="core", module="soft_sim.cli.prepare_mesh",
                         help="Generate the conforming tetrahedral finger mesh"),
    "finalize": dict(env="dataset", module="soft_sim.cli.finalize",
                     help="Convert a staged capture into the official LeRobot dataset"),
    "validate": dict(env="core", module="soft_sim.cli.validate",
                     help="Organize data, camera, FEM and collision checks for one run"),
    "preview": dict(env="dataset", module="soft_sim.cli.preview",
                    help="Build derived previews from a run or dataset"),
    "download-assets": dict(env="core", module="soft_sim.cli.download_assets",
                            help="Download pinned external input assets"),
    "setup-sources": dict(env="core", module="soft_sim.cli.setup_sources",
                          help="Prepare pinned third-party source snapshots"),
    "test": dict(env="core", module="soft_sim.cli.test",
                 help="Run a test group in its registered environment"),
    "migrate": dict(env="core", module="soft_sim.cli.migrate",
                    help="List or copy pre-refactor outputs into the storage root (dry run by default)"),
    "doctor": dict(env="core", module="soft_sim.cli.doctor",
                   help="Report interpreters, packages, sources and paths"),
}

ALIASES = {
    "export_scene": "export-scene",
    "check_installation": "check-installation",
    "prepare_mesh": "prepare-mesh",
    "download_assets": "download-assets",
    "setup_sources": "setup-sources",
}


def resolve(name):
    return ALIASES.get(name, name)


def command(name):
    name = resolve(name)
    if name not in COMMANDS:
        raise KeyError(name)
    return COMMANDS[name]
