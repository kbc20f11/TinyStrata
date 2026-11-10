"""The real model-switch transaction preserves text when its image worker fails.

Only bundle creation and process objects are fixtures. No HTTP listener, native
process, model artifact, GPU, download or installed profile is used.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from serve import frontend
from serve.frontend import ChatTemplate
from serve.model_profiles import ModelProfile
from serve.model_switch import ModelBundle, ModelSwitcher
from serve.server import ByteTokenizer, MockEngine, Service, make_handler


ROOT = Path(__file__).resolve().parents[1]
OLD = "qwen3.6-35b-a3b"
NEW = "gemma-4-26b-a4b-it"


class Engine(MockEngine):
    def __init__(self, tok, events, name, *, loaded=False, fail=False):
        super().__init__(tok, name + " text works", max_context=8192)
        self.events, self.name, self.running, self.fail = events, name, loaded, fail
        self.info = {"vision": 1}

    def alive(self):
        return self.running

    def restart(self, tries=1):
        self.events.append(self.name + ".engine.restart")
        if self.fail:
            raise RuntimeError("candidate language engine refused")
        self.running = True

    def unload(self):
        self.events.append(self.name + ".engine.unload")
        self.running = False

    def close(self):
        self.events.append(self.name + ".engine.close")
        self.running = False

    def generate(self, *args, **kwargs):
        if not self.running:
            raise AssertionError("a stopped fixture language engine cannot answer")
        yield from super().generate(*args, **kwargs)


class Vision:
    def __init__(self, events, name, *, failed=False):
        self.events, self.name = events, name
        self.gpu, self.elastic, self.running = True, False, False
        self.startup_error = "earlier image worker failure" if failed else ""

    def alive(self):
        return self.running

    def restart(self):
        self.events.append(self.name + ".vision.restart")
        raise RuntimeError(self.name + " image allocation refused")

    def unload(self):
        self.events.append(self.name + ".vision.unload")
        self.running = False

    def shutdown(self):
        self.events.append(self.name + ".vision.shutdown")
        self.running = False


class Download:
    def status(self):
        return {"state": "idle"}

    def close(self):
        pass


class Profiles:
    current_id = OLD

    def __init__(self, root):
        self.rows = {}
        for model, arch, filename in ((OLD, "qwen35moe", "old.json"), (NEW, "gemma4", "new.json")):
            path = root / filename
            cfg = {"arch": arch, "model_name": model, "args": ["--max-context", "8192"],
                   "fixture_setting": "preserve"}
            path.write_text(json.dumps(cfg), encoding="utf-8")
            self.rows[model] = ModelProfile(model, arch, path, cfg)

    def resolve(self, model):
        if model not in self.rows:
            raise ValueError("unknown fixture model")
        return model

    def get(self, model):
        return self.rows[model]

    def context_info(self, model):
        self.resolve(model)
        return {"native_tokens": 8192, "allocated_tokens": None, "max_tokens": 8192}

    def list(self):
        return [{"id": profile.id, "arch": profile.arch, "installed": True, "vision": True}
                for profile in self.rows.values()]


class ModelSwitchVisionResilience(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tinystrata-switch-image-test-")
        self.addCleanup(temporary.cleanup)
        self.profiles = Profiles(Path(temporary.name))
        self.original_profiles = {name: p.path.read_bytes() for name, p in self.profiles.rows.items()}
        self.events = []
        self.tok = ByteTokenizer()
        self.old_engine = Engine(self.tok, self.events, "old", loaded=True)
        self.old_vision = Vision(self.events, "old", failed=True)
        self.svc = Service(self.old_engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                           model_name=OLD, vision=self.old_vision)
        self.svc.downloads = Download()
        api = SimpleNamespace(responses_api=SimpleNamespace(_kept_prompt_tools=(None, None)))
        with mock.patch("serve.model_switch.ModelProfiles", return_value=self.profiles):
            self.control = ModelSwitcher(self.svc, self.profiles.get(OLD).config,
                                         self.profiles.get(OLD).path, api)
        self.addCleanup(self.control.close)
        self.control._downloads[NEW] = Download()
        self.svc.models = self.control
        # Publication uses the real frontend global policy, restored for any
        # other test sharing this Python process.
        env = mock.patch.object(frontend, "ENV_TO_USER", frontend.ENV_TO_USER)
        env.start(); self.addCleanup(env.stop)
        with frontend._TOOL_ORDERS_LOCK:
            previous_orders = dict(frontend._TOOL_ORDERS)
        def restore_orders():
            with frontend._TOOL_ORDERS_LOCK:
                frontend._TOOL_ORDERS.clear()
                frontend._TOOL_ORDERS.update(previous_orders)
        self.addCleanup(restore_orders)

    def switch(self, *, language_fails=False):
        self.new_engine = Engine(self.tok, self.events, "new", fail=language_fails)
        self.new_vision = Vision(self.events, "new")
        fresh = Service(self.new_engine, self.tok, self.svc.template, model_name=NEW, vision=self.new_vision)
        profile = self.profiles.get(NEW)
        bundle = ModelBundle(NEW, deepcopy(profile.config), profile.path, fresh)
        # Run public admission, the real worker transaction, _activate,
        # publication and rollback. No activation helper is replaced.
        with mock.patch.object(self.control, "_prepare", return_value=bundle) as prepare, redirect_stdout(StringIO()):
            self.control.start(NEW)
            worker = self.control._worker
            self.assertIsNotNone(worker)
            worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
        prepare.assert_called_once_with(NEW)
        return self.control.status()

    def assert_profiles_unchanged(self):
        for model, profile in self.profiles.rows.items():
            self.assertEqual(profile.path.read_bytes(), self.original_profiles[model])

    def assert_text_works(self, prefix):
        messages = [{"role": "user", "content": "hello"}]
        ids, _, _ = self.svc.prepare(messages, None, {}, max_new=128)
        result = list(self.svc.engine.generate(ids, 128, {}, self.svc.stopping))
        self.assertTrue(self.tok.decode(result).startswith(prefix + " text works"))

    def test_candidate_image_failure_publishes_usable_new_text_model(self):
        status = self.switch()
        self.assertEqual((status["state"], status["phase"], status["model"]), ("ready", "ready", NEW))
        self.assertIsNone(status["error"])
        self.assertIs(self.svc.engine, self.new_engine)
        self.assertIs(self.svc.vision, self.new_vision)
        self.assertTrue(self.svc.loaded())
        self.assertFalse(self.svc.vision_available())
        self.assertIn("new image allocation refused", self.svc.vision_error)
        self.assertLess(self.events.index("new.vision.restart"), self.events.index("new.engine.restart"))
        self.assertIn("old.engine.close", self.events)
        self.assertFalse(self.old_engine.alive())
        self.assert_text_works("new")
        self.assert_profiles_unchanged()

    def test_candidate_language_failure_restores_old_text_despite_old_image_failure(self):
        status = self.switch(language_fails=True)
        self.assertEqual(status["state"], "error")
        self.assertTrue(status["rolled_back"])
        self.assertIn("candidate language engine refused", status["error"])
        self.assertNotIn("old model restoration failed", status["error"])
        self.assertEqual(status["model"], OLD)
        self.assertIs(self.svc.engine, self.old_engine)
        self.assertIs(self.svc.vision, self.old_vision)
        self.assertTrue(self.svc.model_available)
        self.assertTrue(self.svc.loaded())
        self.assertFalse(self.svc.vision_available())
        self.assertIn("old image allocation refused", self.svc.vision_error)
        self.assertLess(self.events.index("new.engine.close"), self.events.index("old.engine.restart"))
        self.assertLess(self.events.index("old.vision.restart"), self.events.index("old.engine.restart"))
        self.assertFalse(self.new_engine.alive())
        self.assert_text_works("old")
        self.assert_profiles_unchanged()

    def test_models_route_reports_active_image_failure_without_changing_inactive_capability(self):
        self.switch()
        handler = object.__new__(make_handler(self.svc))
        handler.headers, handler.path = {}, "/v1/models"
        handler._json = lambda status, value: setattr(handler, "result", (status, value))
        # File-management inventory is unrelated to live image capability;
        # replace its disk metadata reads, not the real catalog/HTTP decision.
        with mock.patch("serve.model_library.inventory", side_effect=lambda *_: SimpleNamespace(
                management={"can_delete": False, "can_download": False})):
            handler.do_GET()
        status, response = handler.result
        self.assertEqual(status, 200)
        models = {row["id"]: row for row in response["data"]}
        self.assertEqual(models[NEW]["architecture"]["input_modalities"], ["text"])
        self.assertEqual(models[OLD]["architecture"]["input_modalities"], ["text", "image"])
        self.assertTrue(models[NEW]["active"])
        self.assertFalse(models[OLD]["active"])
        self.assert_profiles_unchanged()


if __name__ == "__main__":
    unittest.main()
