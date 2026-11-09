import contextlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "hard-negative-colab.ipynb"
RUNBOOK_PATH = ROOT / "docs" / "hard-negative-colab-runbook.md"
GUARD_START = "# --- BEGIN COLAB MEDIA GUARD ---"
GUARD_END = "# --- END COLAB MEDIA GUARD ---"
MEDIA_PREFIXES = ("torchvision", "torchaudio")


@contextlib.contextmanager
def without_media_modules():
    """Temporarily hide any media modules imported by the host test process."""
    saved = {}
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in MEDIA_PREFIXES):
            saved[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        sys.modules.update(saved)


class HardNegativeColabEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = [
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.source = "\n".join(cls.cells)
        cls.runbook = RUNBOOK_PATH.read_text(encoding="utf-8")

    def cell(self, marker):
        return next(cell for cell in self.cells if marker in cell)

    def guard_namespace(self):
        cell3 = self.cell("# CELL 3")
        start = cell3.index(GUARD_START)
        end = cell3.index(GUARD_END) + len(GUARD_END)
        namespace = {}
        exec(compile(cell3[start:end], "cell3-media-guard", "exec"), namespace)
        return namespace

    def test_notebook_header_identifies_the_current_guard_build(self):
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("Environment guard build: `hardneg-env-v3`", markdown)

    def test_cell2_removes_optional_media_before_and_after_install(self):
        cell2 = self.cell("# CELL 2")
        uninstall = "%pip uninstall -y torchvision torchaudio"
        install = (
            "%pip install -q torch==2.6.0 "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        first_uninstall = cell2.index(uninstall)
        install_at = cell2.index(install)
        second_uninstall = cell2.index(uninstall, install_at)
        self.assertLess(first_uninstall, install_at)
        self.assertLess(install_at, second_uninstall)
        self.assertIn("ENV_GUARD_BUILD = 'hardneg-env-v3'", cell2)
        self.assertIn("survived Cell 2", cell2)
        self.assertIn("CELL2_INSTALL_OK = True", cell2)

    def test_cell3_checks_metadata_submodules_and_transformers_cache(self):
        cell3 = self.cell("# CELL 3")
        self.assertIn("importlib.metadata.version(package)", cell3)
        self.assertIn("name.startswith(package + '.')", cell3)
        self.assertIn("assert_transformers_media_state", cell3)
        self.assertIn("sys.modules.get('transformers.image_utils')", cell3)
        self.assertIn("import transformers.image_utils", cell3)
        self.assertIn("DebertaV2ForSequenceClassification", cell3)
        self.assertIn("CELL3_GUARD_OK = True", cell3)
        self.assertIn("guard build", cell3)
        self.assertLess(
            self.source.index("# CELL 3"), self.source.index("# CELL 4")
        )

    def test_notebook_does_not_ship_stale_cell_outputs(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell.get("execution_count"), index)
            self.assertEqual(cell.get("outputs"), [], index)

    def test_gate0_and_single_run_require_same_runtime_guard(self):
        cell4 = self.cell("# CELL 4")
        cell5 = self.cell("# CELL 5")
        for cell in (cell4, cell5):
            self.assertIn("ENV_GUARD_BUILD", cell)
            self.assertIn("CELL3_GUARD_OK", cell)
        self.assertIn("CELL2_INSTALL_OK", cell4)

    def test_media_guard_passes_only_for_a_clean_interpreter(self):
        guard = self.guard_namespace()
        with without_media_modules(), mock.patch.object(
            importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError
        ), mock.patch.object(importlib.util, "find_spec", return_value=None):
            guard["assert_media_absent"]("clean-regression")

    def test_media_guard_catches_loaded_torchvision_submodule(self):
        guard = self.guard_namespace()
        with without_media_modules(), mock.patch.dict(
            sys.modules, {"torchvision._C": object()}, clear=False
        ), mock.patch.object(
            importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError
        ), mock.patch.object(importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(AssertionError, r"torchvision\._C"):
                guard["assert_media_absent"]("stale-submodule-regression")

    def test_media_guard_catches_installed_distribution_without_top_level_match(self):
        guard = self.guard_namespace()
        with without_media_modules(), mock.patch.object(
            importlib.metadata, "version", return_value="0.20.1"
        ), mock.patch.object(importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(AssertionError, "distribution"):
                guard["assert_media_absent"]("metadata-regression")

    def test_media_guard_catches_find_spec_match(self):
        guard = self.guard_namespace()
        fake_spec = types.SimpleNamespace(origin="C:/fake/torchvision/__init__.py")
        with without_media_modules(), mock.patch.object(
            importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError
        ), mock.patch.object(
            importlib.util,
            "find_spec",
            side_effect=lambda name: fake_spec if name == "torchvision" else None,
        ):
            with self.assertRaisesRegex(AssertionError, "torchvision"):
                guard["assert_media_absent"]("spec-regression")

    def test_media_guard_fails_closed_when_spec_lookup_is_broken(self):
        guard = self.guard_namespace()
        with without_media_modules(), mock.patch.dict(
            sys.modules, {"torchvision": object()}, clear=False
        ), mock.patch.object(
            importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError
        ), mock.patch.object(
            importlib.util,
            "find_spec",
            side_effect=ValueError("module has no spec"),
        ):
            with self.assertRaisesRegex(AssertionError, "broken import spec"):
                guard["assert_media_absent"]("broken-spec-regression")

    def test_guard_catches_transformers_cached_availability(self):
        guard = self.guard_namespace()
        fake_transformers = types.SimpleNamespace(
            is_torchvision_available=lambda: True,
            utils=types.SimpleNamespace(is_torchaudio_available=lambda: False),
        )
        with self.assertRaisesRegex(AssertionError, "Transformers cached optional media"):
            guard["assert_transformers_media_state"](fake_transformers, None)

    def test_guard_catches_transformers_utils_cached_torchaudio(self):
        guard = self.guard_namespace()
        fake_transformers = types.SimpleNamespace(
            is_torchvision_available=lambda: False,
            utils=types.SimpleNamespace(is_torchaudio_available=lambda: True),
        )
        with self.assertRaisesRegex(AssertionError, "torchaudio"):
            guard["assert_transformers_media_state"](fake_transformers, None)

    def test_guard_catches_stale_transformers_image_utils_reference(self):
        guard = self.guard_namespace()
        fake_transformers = types.SimpleNamespace(
            is_torchvision_available=lambda: False,
            is_torchaudio_available=lambda: False,
        )
        fake_image_utils = types.SimpleNamespace(torchvision_io=object())
        with self.assertRaisesRegex(AssertionError, "stale references"):
            guard["assert_transformers_media_state"](
                fake_transformers, fake_image_utils
            )

    def test_guard_stops_the_exact_torchvision_operator_failure(self):
        venv_python = ROOT / ".venv-deberta" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        if not venv_python.is_file():
            self.skipTest("frozen .venv-deberta is not available for the integration replay")

        prefix = textwrap.dedent(
            """
            import importlib.metadata
            import importlib.util
            import pathlib
            import shutil
            import sys
            import tempfile

            root = pathlib.Path(tempfile.mkdtemp(prefix="promptshield-torchvision-replay-"))

            torchvision = root / "torchvision"
            torchvision.mkdir()
            (torchvision / "__init__.py").write_text(
                'raise RuntimeError("operator torchvision::nms does not exist")\\n',
                encoding="utf-8",
            )
            torchvision_dist = root / "torchvision-999.0.dist-info"
            torchvision_dist.mkdir()
            (torchvision_dist / "METADATA").write_text(
                "Metadata-Version: 2.1\\nName: torchvision\\nVersion: 999.0\\n",
                encoding="utf-8",
            )

            pil = root / "PIL"
            pil.mkdir()
            (pil / "__init__.py").write_text("__version__ = '9.1.0'\\n", encoding="utf-8")
            (pil / "Image.py").write_text(
                "class Resampling:\\n"
                "    NEAREST=0\\n"
                "    BOX=1\\n"
                "    BILINEAR=2\\n"
                "    HAMMING=3\\n"
                "    BICUBIC=4\\n"
                "    LANCZOS=5\\n"
                "class Image: pass\\n",
                encoding="utf-8",
            )
            (pil / "ImageOps.py").write_text("pass\\n", encoding="utf-8")
            pil_dist = root / "Pillow-999.0.dist-info"
            pil_dist.mkdir()
            (pil_dist / "METADATA").write_text(
                "Metadata-Version: 2.1\\nName: Pillow\\nVersion: 999.0\\n",
                encoding="utf-8",
            )

            sys.path.insert(0, str(root))
            """
        )
        suffix = textwrap.dedent(
            """

            import transformers

            assert transformers.is_torchvision_available() is True
            try:
                assert_transformers_media_state(transformers, None)
            except AssertionError as error:
                assert "Transformers cached optional media" in str(error), str(error)
            else:
                raise SystemExit("guard did not stop cached TorchVision availability")

            try:
                import transformers.image_utils
            except RuntimeError as error:
                assert "operator torchvision::nms does not exist" in str(error), str(error)
            else:
                raise SystemExit("the exact TorchVision operator failure was not reproduced")

            shutil.rmtree(root, ignore_errors=True)
            print("guard stopped the exact torchvision::nms failure before image_utils import")
            """
        )
        script = prefix + self.guard_block_text() + suffix
        completed = subprocess.run(
            [str(venv_python), "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(completed.stdout + completed.stderr)[-4000:],
        )
        self.assertIn(
            "guard stopped the exact torchvision::nms failure",
            completed.stdout,
        )

    def guard_block_text(self):
        cell3 = self.cell("# CELL 3")
        start = cell3.index(GUARD_START)
        end = cell3.index(GUARD_END) + len(GUARD_END)
        return cell3[start:end]

    def test_runbook_documents_the_guard_blind_spots_and_reset(self):
        self.assertIn("operator torchvision::nms does not exist", self.runbook)
        self.assertIn("torchvision._C", self.runbook)
        self.assertIn("cached", self.runbook)
        self.assertIn("guard build hardneg-env-v3", self.runbook)
        self.assertIn("Runtime → Disconnect and delete runtime", self.runbook)
        self.assertIn("do not change the torch version", self.runbook)

    def test_repository_source_does_not_import_removed_media_packages(self):
        for path in (ROOT / "src").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("import torchvision", source, path)
            self.assertNotIn("import torchaudio", source, path)


if __name__ == "__main__":
    unittest.main()
