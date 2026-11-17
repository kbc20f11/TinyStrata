"""Retired image generation has no UI entrance; image input and tool images remain.

Node executes the production navigation and busy-state helpers. Browser visual
and attachment workflows are verified separately through the running UI.
"""
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "ui/web"
NODE = shutil.which("node")


class UiElements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.tabs = set(), []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if attrs.get("role") == "tab":
            self.tabs.append(attrs.get("data-tab"))


class ImageGenerationUiRemoval(unittest.TestCase):
    def test_only_current_tabs_remain_and_vision_controls_are_preserved(self):
        elements = UiElements((WEB / "index.html").read_text(encoding="utf-8"))
        self.assertEqual(elements.tabs, ["chat", "models", "monitor", "about"])
        self.assertTrue({"attach-btn", "input", "composer-model", "sidebar-search"} <= elements.ids)
        self.assertFalse(any(name.startswith(("img-", "view-images", "tab-btn-images")) for name in elements.ids))
        app = (WEB / "app.js").read_text(encoding="utf-8")
        for retired in ("images/generations", "image_generation", "imageBusy", "imageLive", "img-form"):
            self.assertNotIn(retired, app)
        for retained in ("canUseModelImages", "composer.attachPictures", "health.images", "StrataLocalTools.safeImage"):
            self.assertIn(retained, app)
        self.assertIn('tr("about.imageInput", "Image input")', app)
        css = (WEB / "app.css").read_text(encoding="utf-8")
        self.assertIn(".msg-images img", css)
        self.assertIn(".tool-image img", css)
        self.assertNotIn(".img-results", css)

    def test_each_locale_removes_generation_copy_and_retains_image_input_copy(self):
        retired = {"tab.images", "pill.image", "model.imageBusy", "model.imageLoading", "model.imageLoadingDot",
                   "monitor.generatingImage", "monitor.imageNote"}
        for path in (WEB / "locales").glob("*.json"):
            with self.subTest(locale=path.stem):
                strings = json.loads(path.read_text(encoding="utf-8"))
                self.assertFalse(retired & strings.keys())
                self.assertFalse(any(key.startswith("images.") for key in strings))
                for retained in ("about.imageInput", "composer.attachPictures", "file.picturesOff"):
                    self.assertTrue(strings[retained])

    @unittest.skipUnless(NODE, "Node.js is required for UI helper checks")
    def test_legacy_images_hash_returns_home_and_current_tabs_still_work(self):
        source = (WEB / "app.js").read_text(encoding="utf-8")
        navigation = source[source.index('let tab = "chat";'):source.index('for (const b of document.querySelectorAll(".st-tab")) b.onclick')]
        script = r'''
const assert = require("node:assert/strict");
const buttons = ["chat", "models", "monitor", "about"].map((name) => ({dataset: {tab: name}, setAttribute(key, value) {this[key] = value;}}));
const nodes = new Map(["input", "view-chat", "view-models", "view-monitor", "view-about", "cfg-card"].map((id) => [id, {id, hidden: false, focus() {this.focused = true;}}]));
const $ = (id) => {assert.ok(nodes.has(id), "unexpected retired element: " + id); return nodes.get(id);};
const location = {hash: "#images", pathname: "/"};
const history = {replaceState(a, b, url) {location.hash = url.startsWith("#") ? url : "";}};
const document = {querySelectorAll: () => buttons}, requestAnimationFrame = (fn) => fn();
let lastMetrics = null, mcpLoads = 0, catalogLoads = 0;
const loadMcp = () => mcpLoads++, renderModelLibrary = () => catalogLoads++, refreshModelCatalog = () => catalogLoads++, render = () => {};
''' + navigation + r'''
showTab("images");
assert.equal(tab, "chat"); assert.equal(location.hash, ""); assert.equal($("view-chat").hidden, false);
assert.equal($("input").focused, true);
for (const name of ["models", "monitor", "about", "chat"]) {
  showTab(name);
  assert.equal(tab, name);
  assert.equal(buttons.filter((button) => button["aria-selected"] === "true").length, 1);
  for (const view of ["chat", "models", "monitor", "about"]) assert.equal($("view-" + view).hidden, view !== name);
}
assert.equal(mcpLoads, 1); assert.equal(catalogLoads, 2); assert.equal($("cfg-card").hidden, true);
'''
        result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(NODE, "Node.js is required for UI helper checks")
    def test_chat_and_resource_busy_states_still_protect_unload(self):
        source = (WEB / "app.js").read_text(encoding="utf-8")
        helpers = source[source.index("let modelOp = null;"):source.index('$("unload-btn").onclick')]
        script = r'''
const assert = require("node:assert/strict");
let busy = null, resourceOperation = null, lastMetrics = null, modelSwitching = false, shown;
const tr = (key, fallback) => fallback, render = () => {};
''' + helpers + r'''
setModelButton = (...args) => {shown = args;};
modelButton({state:"idle"}); assert.equal(shown[0], "Unload"); assert.equal(shown[2], false);
busy = {}; modelButton({state:"unloaded"}); assert.equal(shown[0], "Loading…"); assert.equal(shown[2], true);
busy = null; resourceOperation = "compact"; modelButton({state:"idle"}); assert.equal(shown[2], true);
resourceOperation = null; modelButton({state:"generating"}); assert.equal(shown[2], true);
modelButton({state:"idle", phase:"generating an image"}); assert.equal(shown[2], false);
modelButton({state:"unloaded"}); assert.equal(shown[0], "Load"); assert.equal(shown[2], false);
'''
        result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
