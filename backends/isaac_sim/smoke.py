"""Check Isaac Sim startup in its own environment; first run prompts for EULA."""

from isaacsim import SimulationApp

# Initialize Kit before importing other Omniverse/Isaac simulation modules.
app = SimulationApp({"headless": True})
try:
    if not app.is_running():
        raise RuntimeError("Isaac Sim did not start.")
    for _ in range(10):
        app.update()
    if not app.is_running():
        raise RuntimeError("Isaac Sim stopped during startup verification.")
    print("Isaac Sim startup OK", flush=True)
finally:
    app.close()
