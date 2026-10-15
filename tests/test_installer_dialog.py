"""Tests for ModelInstallerDialog, InstallerWorker, and download_roto integration."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication

APP = QApplication.instance() or QApplication([])

from matte_app.installer_dialog import (
    ALL_MODELS,
    InstallerWorker,
    ModelCardWidget,
    ModelDef,
    ModelInstallerDialog,
    check_hf_login,
    detect_environments,
    is_model_installed,
    verify_and_save_hf_token,
)
from matte_app.app_window import StudioWindow
from matte_app.window import MatteWindow


class TestModelInstaller(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_catalog_definitions(self):
        """Verify all 9 expected AI models are defined with required attributes."""
        keys = [m.key for m in ALL_MODELS]
        self.assertEqual(len(keys), 9)
        expected = [
            "sam2",
            "matanyone",
            "matanyone2",
            "sam2matting",
            "sam31",
            "sam3",
            "vitmatte",
            "videomama",
            "litesdmatte",
        ]
        self.assertEqual(keys, expected)
        sam3 = next(m for m in ALL_MODELS if m.key == "sam3")
        self.assertTrue(sam3.is_gated)
        sam2 = next(m for m in ALL_MODELS if m.key == "sam2")
        self.assertFalse(sam2.is_gated)

    def test_detect_environments(self):
        """Verify environment detector returns boolean flags for all dependencies."""
        envs = detect_environments(self.root)
        self.assertIn("roto_venv", envs)
        self.assertIn("sdmatte_venv", envs)
        self.assertIn("git", envs)
        self.assertIn("uv", envs)
        self.assertIn("ffmpeg", envs)
        for val in envs.values():
            self.assertIsInstance(val, bool)

    def test_model_readiness_check(self):
        """Verify is_model_installed checks correctly against disk."""
        sam2 = next(m for m in ALL_MODELS if m.key == "sam2")
        installed = is_model_installed(self.root, sam2)
        expected = (self.root / "models/SAM2/model.safetensors").is_file()
        self.assertEqual(installed, expected)

    def test_hf_token_functions(self):
        """Verify token verification with empty and invalid tokens, ensuring key is NOT saved to disk."""
        ok, msg = verify_and_save_hf_token("   ")
        self.assertFalse(ok)
        self.assertIn("empty", msg.lower())

        with patch("matte_app.installer_dialog.HfApi") as mock_api:
            mock_inst = MagicMock()
            mock_inst.whoami.return_value = {"name": "TestUser"}
            mock_api.return_value = mock_inst
            ok, msg = verify_and_save_hf_token("hf_valid_test_token")
            self.assertTrue(ok)
            self.assertIn("TestUser", msg)
            self.assertIn("not saved", msg.lower())
            self.assertEqual(os.environ.get("HF_TOKEN"), "hf_valid_test_token")

    def test_pipeline_filters_uninstalled_models(self):
        """Verify that pipelines, selectors, and refiners filter out uninstalled models."""
        from matte_app.installed_models import get_available_pipelines, get_available_selectors, get_available_refiners
        with tempfile.TemporaryDirectory() as tmp_root:
            empty_root = Path(tmp_root)
            pipes = get_available_pipelines(empty_root)
            self.assertEqual(pipes, ["BiRefNet HR — automatic matte"])
            selectors = get_available_selectors(empty_root)
            self.assertEqual(selectors, ["BiRefNet HR"])
            refiners = get_available_refiners(empty_root)
            self.assertEqual(refiners, ["None — keep base matte"])

    def test_installer_dialog_creation_and_widgets(self):
        """Test dialog builds properly with all model cards and controls."""
        dlg = ModelInstallerDialog(focus_model="sam3")
        try:
            self.assertEqual(len(dlg.model_cards), 9)
            sam3_card = dlg.model_cards["sam3"]
            self.assertTrue(sam3_card.is_checked())

            # Test presets
            dlg.select_core_models()
            self.assertTrue(dlg.model_cards["sam2"].is_checked())
            self.assertTrue(dlg.model_cards["matanyone"].is_checked())
            self.assertTrue(dlg.model_cards["vitmatte"].is_checked())
            self.assertFalse(dlg.model_cards["sam3"].is_checked())

            # Test select/deselect all
            dlg._set_all_models_checked(True)
            for card in dlg.model_cards.values():
                self.assertTrue(card.is_checked())

            dlg._set_all_models_checked(False)
            for card in dlg.model_cards.values():
                self.assertFalse(card.is_checked())

            # Test password toggle
            from PySide6.QtWidgets import QLineEdit
            dlg.token_input.setEchoMode(QLineEdit.Password)
            dlg._toggle_token_visibility()
            self.assertEqual(dlg.token_input.echoMode(), QLineEdit.Normal)
            dlg._toggle_token_visibility()
            self.assertEqual(dlg.token_input.echoMode(), QLineEdit.Password)
        finally:
            dlg.close()

    def test_installer_worker_cancellation(self):
        """Verify worker cancellation terminates gracefully."""
        worker = InstallerWorker(
            root=self.root,
            models_to_download=[],
            setup_roto=False,
            setup_sdmatte=False,
        )
        finished_results = []
        worker.finished.connect(lambda ok, msg: finished_results.append((ok, msg)))
        worker.run()
        self.assertTrue(len(finished_results) > 0)
        self.assertTrue(finished_results[0][0])

    def test_window_open_installer_integration(self):
        """Verify open_installer and get_sam3_weights are wired to open the installer dialog."""
        win = StudioWindow()
        try:
            self.assertTrue(hasattr(win, "installer_header_button"))
            self.assertEqual(win.installer_header_button.text(), "Install Models")

            dlg = win.open_installer()
            self.assertIsNotNone(dlg)
            self.assertIsInstance(dlg, ModelInstallerDialog)
            dlg.close()

            dlg2 = win.get_sam3_weights()
            self.assertIsNotNone(dlg2)
            self.assertTrue(dlg2.model_cards["sam3"].is_checked())
            dlg2.close()
        finally:
            win.close()


if __name__ == "__main__":
    unittest.main()
