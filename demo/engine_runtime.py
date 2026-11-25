"""Isolate ViZDoom in an ASCII temporary path (Windows Unicode path workaround)."""
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


@contextmanager
def engine_runtime():
    source = ROOT / "vendor" / "vizdoom"
    if not source.exists():
        raise RuntimeError("Install dependencies into demo/vendor; see README.md")
    previous_cwd = Path.cwd()
    # Windows cannot unlink a loaded .pyd until this Python process exits.
    with tempfile.TemporaryDirectory(prefix="vizdoom_demo_", ignore_cleanup_errors=True) as temporary:
        runtime = Path(temporary).resolve()
        assert runtime.parent == Path(tempfile.gettempdir()).resolve()
        if not str(runtime).isascii():
            raise RuntimeError("This Windows ViZDoom wheel needs an ASCII runtime path")
        package = runtime / "vizdoom"
        package.mkdir()
        for file in source.iterdir():
            if file.is_file():
                shutil.copy2(file, package / file.name)
        scenarios = package / "scenarios"
        scenarios.mkdir()
        for name in ("predict_position.cfg", "predict_position.wad"):
            shutil.copy2(source / "scenarios" / name, scenarios / name)
        sys.path.insert(0, str(ROOT / "vendor"))
        sys.path.insert(0, str(runtime))
        os.chdir(runtime)
        try:
            yield importlib.import_module("vizdoom")
        finally:
            os.chdir(previous_cwd)
            sys.path.remove(str(runtime))
            sys.path.remove(str(ROOT / "vendor"))
