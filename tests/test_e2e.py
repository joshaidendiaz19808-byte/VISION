"""The real app in real browsers: iPhone-sized WebKit (Safari's engine) and desktop Chromium.
Every flow a user can take, failing on any JavaScript error or horizontal scroll on a phone."""
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uvicorn

pw = pytest.importorskip("playwright.sync_api")

from api import server  # noqa: E402
from checks import connectors  # noqa: E402

SHOTS = Path(os.environ.get("VISION_SHOTS", "/tmp/vision_shots"))


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(server.app, port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(5)


@pytest.fixture(scope="module", params=["iphone-webkit", "desktop-chromium"])
def browser_page(request, base_url):
    with pw.sync_playwright() as p:
        if request.param == "iphone-webkit":
            browser = p.webkit.launch()
            ctx = browser.new_context(**p.devices["iPhone 13"])
        else:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        # expected API refusals (400s the app turns into messages) log "Failed to load resource" - not a bug
        page.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text
                and errors.append(m.text))
        page.goto(base_url)
        yield request.param, page, errors
        browser.close()


def fresh(page, base_url):
    page.goto("about:blank")
    page.goto(base_url)
    page.wait_for_selector("#run")


def verdict(page):
    page.wait_for_selector(".verdict .big", timeout=30000)
    return page.inner_text(".verdict .big")


def no_horizontal_scroll(page):
    return page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


def test_sample_flow(browser_page, base_url):
    name, page, errors = browser_page
    fresh(page, base_url)
    page.click("#sample")
    v = verdict(page)
    assert v in ("LOOKS REAL", "QUESTIONABLE", "LOOKS FAKE")
    assert page.is_visible("#options") and "trading" in page.inner_text("#options h3")
    # change a setting and re-check
    page.fill("#options input[type=number] >> nth=1", "500")
    page.click("#run")
    page.wait_for_function("document.querySelector('#verdict').innerText.includes('Tried many versions')",
                           timeout=30000)
    assert no_horizontal_scroll(page)
    SHOTS.mkdir(exist_ok=True)
    page.screenshot(path=str(SHOTS / f"{name}-sample.png"), full_page=True)
    assert not errors, errors


def test_every_kind_switches_cleanly(browser_page, base_url):
    name, page, errors = browser_page
    fresh(page, base_url)
    for kind in ("trading", "ab", "ml", "data", "code", "auto"):
        page.click(f'#kinds button[data-kind="{kind}"]')
        assert page.is_hidden("#results-card") == (kind == "code")
        assert page.is_visible("#ml-files") == (kind == "ml")
        assert no_horizontal_scroll(page)
    assert not errors, errors


def test_ab_upload(browser_page, base_url, tmp_path):
    name, page, errors = browser_page
    fresh(page, base_url)
    f = tmp_path / "ab.csv"
    pd.DataFrame({"variant": ["control"] * 4000 + ["B"] * 3600,
                  "converted": [1] * 400 + [0] * 3600 + [1] * 450 + [0] * 3150}).to_csv(f, index=False)
    page.set_input_files("#file", str(f))
    page.click("#run")
    assert verdict(page) == "LOOKS FAKE"  # 4000 vs 3600 is a broken split
    assert "Groups split as planned" in page.inner_text("#verdict")
    assert page.input_value("#options select >> nth=3") == "control"
    page.screenshot(path=str(SHOTS / f"{name}-ab.png"), full_page=True)
    assert not errors, errors


def test_ml_leak(browser_page, base_url, tmp_path):
    name, page, errors = browser_page
    fresh(page, base_url)
    page.click('#kinds button[data-kind="ml"]')
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f": rng.normal(size=600)})
    df["y"] = (df.f + rng.normal(size=600) > 0).astype(int)
    df["refund"] = df["y"]
    df[:400].to_csv(tmp_path / "tr.csv", index=False)
    df[400:].to_csv(tmp_path / "te.csv", index=False)
    page.set_input_files("#ml-train", str(tmp_path / "tr.csv"))
    page.set_input_files("#ml-test", str(tmp_path / "te.csv"))
    page.click("#run")
    assert verdict(page) == "LOOKS FAKE"
    assert "refund" in page.inner_text("#verdict")
    assert not errors, errors


def test_code_audit_combined_and_ai_needs_key(browser_page, base_url, tmp_path):
    name, page, errors = browser_page
    fresh(page, base_url)
    page.click("#sample")
    verdict(page)
    code = tmp_path / "strategy.py"
    code.write_text('import pandas as pd\ndf = pd.read_csv("p.csv")\ndf["sig"] = df["close"].shift(-1) > df["close"]\n')
    page.click("#code-title")
    page.set_input_files("#code-files", str(code))
    page.click("#run-code")
    page.wait_for_function("document.querySelector('.verdict .sub').textContent.includes('code behind')")
    assert verdict(page) == "LOOKS FAKE"
    assert "strategy.py:3" in page.inner_text("#verdict")
    page.click("#run-ai")
    page.wait_for_selector("#status.error")
    assert "Anthropic" in page.inner_text("#status")
    page.screenshot(path=str(SHOTS / f"{name}-code.png"), full_page=True)
    assert not errors, errors


def test_connect_errors_are_shown(browser_page, base_url):
    name, page, errors = browser_page
    fresh(page, base_url)
    page.click('#source button[data-src="connect"]')
    page.select_option("#provider", "url")
    page.fill("#url", "http://169.254.169.254/latest/meta-data")
    page.click("#run")
    page.wait_for_selector("#status.error")
    assert "private" in page.inner_text("#status")
    page.select_option("#provider", "alpaca")
    page.click("#run")
    page.wait_for_selector("#status.error")
    assert "Fill in" in page.inner_text("#status")
    assert not errors, errors


def test_installable(browser_page, base_url):
    name, page, errors = browser_page
    fresh(page, base_url)
    manifest = page.evaluate("fetch('manifest.webmanifest').then(r => r.json())")
    assert manifest["display"] == "standalone" and len(manifest["icons"]) == 3
    if name == "desktop-chromium":  # WebKit doesn't expose SW registration in headless mode
        page.wait_for_function("navigator.serviceWorker.controller !== null || navigator.serviceWorker.ready", timeout=10000)
        assert page.evaluate("navigator.serviceWorker.ready.then(r => !!r.active)")
    assert not errors, errors


def test_phone_app_mode_cross_origin(browser_page, base_url, tmp_path):
    """Like the phone apps: the screen is served from one origin, the API lives on another."""
    import functools
    import http.server
    import shutil

    name, page, errors = browser_page
    www = tmp_path / "www"
    shutil.copytree(Path(__file__).resolve().parent.parent / "web", www)
    (www / "config.js").write_text(f'window.VISION_API = "{base_url}";\n')
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(www))
    static = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=static.serve_forever, daemon=True).start()
    try:
        page.goto(f"http://localhost:{static.server_address[1]}")
        page.click("#sample")
        page.wait_for_selector(".verdict .big", timeout=30000)
        assert "trades analyzed" in page.inner_text(".verdict .sub")
        assert not errors, errors
    finally:
        static.shutdown()
