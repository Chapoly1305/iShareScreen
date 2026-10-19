"""The ordinary desktop path must not require optional ML dependencies."""
import subprocess
import sys
import textwrap


def test_model_home_is_independent_of_installation(monkeypatch):
    from pathlib import Path
    from isharescreen.enhancement.models import project_home
    monkeypatch.delenv("ISS_SR_HOME", raising=False)
    assert project_home() == Path.home() / ".cache" / "isharescreen" / "super_resolution"


def test_model_home_honors_explicit_override(monkeypatch, tmp_path):
    from isharescreen.enhancement.models import project_home
    monkeypatch.setenv("ISS_SR_HOME", str(tmp_path / "models"))
    assert project_home() == tmp_path / "models"
    monkeypatch.setenv("ISS_SR_HOME", "~/custom-models")
    from pathlib import Path
    assert project_home() == Path.home() / "custom-models"


def test_native_original_path_without_torch_or_opencv():
    # A fresh interpreter catches eager imports hidden by the main test process.
    # Discovery and imports both behave as though the packages were absent.
    code = textwrap.dedent("""
        import importlib.abc
        import importlib.util
        import sys

        blocked = {"torch", "cv2"}
        class NoML(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
        sys.meta_path.insert(0, NoML())
        original_find_spec = importlib.util.find_spec
        def find_spec(name, package=None):
            if name.split(".")[0] in blocked:
                return None
            return original_find_spec(name, package)
        importlib.util.find_spec = find_spec

        from isharescreen.frontend.desktop import app
        from isharescreen.enhancement.live import EnhancementDisplay, EnhancementWorker

        class OriginalRenderer:
            def clear_enhancement(self):
                pass
            def upload_enhanced(self, rgb):
                raise AssertionError("Original bypass attempted enhanced upload")

        worker = EnhancementWorker()
        try:
            state = worker.snapshot()
            assert {o["id"] for o in state["options"] if o["available"]} == {"off"}
            display = EnhancementDisplay(worker, OriginalRenderer(), 8, 8, 1)
            display.update_tile(0, object(), 8)
            worker.select("off")
            assert not display.step()
            assert worker.snapshot()["state"] == "bypass"
            assert worker.snapshot()["submitted"] == 0
            for model in ["bicubic", "edsr", "imdn", "safmn"]:
                try:
                    worker.select(model)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unavailable optional model accepted")
            assert not blocked.intersection(sys.modules)
        finally:
            worker.close()
        assert not worker._thread.is_alive()
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
