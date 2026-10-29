"""Opt-in browser regressions for evaluation-mode settings transitions."""
import os

import pytest

from isharescreen.gui import connect

pytestmark = pytest.mark.skipif(
    os.environ.get("ISS_TEST_BROWSER") != "1", reason="Set ISS_TEST_BROWSER=1 for Chrome interaction tests",
)


@pytest.fixture
def page():
    api = pytest.importorskip("playwright.sync_api")
    html = (connect._FORM.replace("__HEAD__", connect._HEAD)
            .replace("__DEFAULT_AUTH__", "srp")
            .replace("__DECODER_OPTIONS__", '<option value="auto">Auto</option>'
                     '<option value="vt-hevc444" data-codec="hevc">HEVC</option>'
                     '<option value="libav-avc420" data-codec="avc">AVC</option>'))
    with api.sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.set_default_timeout(5000)
        page.route("http://localhost/**", lambda route: route.fulfill(body=html, content_type="text/html"))
        page.goto("http://localhost/")
        yield page
        browser.close()


def settings(page):
    return page.evaluate("""Object.fromEntries(['frontend','codec','chroma','decoder','advertise','scale','audio','curtain'].map(name=>{
      const el=document.querySelector('form').elements[name];return [name,el.type==='checkbox'?el.checked:el.value];
    }))""")


def configure(page):
    page.locator('#advanced > summary').click()
    page.select_option('[name=frontend]', 'desktop')
    page.select_option('[name=codec]', 'avc')
    page.select_option('[name=decoder]', 'libav-avc420')
    page.select_option('[name=advertise]', '1920x1080')
    page.select_option('[name=scale]', '1.5')
    page.uncheck('[name=curtain]')


def test_evaluation_toggle_restores_decoder_and_settings(page):
    configure(page)
    before = settings(page)
    page.check('[name=sr_evaluation]')
    assert settings(page) == dict(frontend='desktop', codec='hevc', chroma='auto', decoder='vt-hevc444',
                                  advertise='960x540', scale='1.0', audio=False, curtain=True)
    assert page.locator('[name=decoder]').is_disabled()
    page.locator('[name=host]').fill('example.test')
    page.uncheck('[name=sr_evaluation]')
    assert settings(page) == before


def test_reset_during_evaluation_keeps_defaults(page):
    configure(page)
    page.check('[name=sr_evaluation]')
    page.click('#reset-settings')
    assert settings(page) == dict(frontend='browser', codec='auto', chroma='auto', decoder='auto',
                                  advertise='auto', scale='2.0', audio=True, curtain=True)
    assert not page.locator('[name=sr_evaluation]').is_checked()
    assert page.locator('[name=decoder]').is_disabled()


def test_reload_during_evaluation_preserves_original_preferences(page):
    configure(page)
    before = settings(page)
    page.check('[name=sr_evaluation]')
    page.reload()
    assert page.locator('[name=sr_evaluation]').is_checked()
    assert page.locator('[name=codec]').is_disabled()
    page.uncheck('[name=sr_evaluation]')
    assert settings(page) == before


def test_browser_decoder_stays_disabled_after_evaluation(page):
    page.check('[name=sr_evaluation]')
    page.uncheck('[name=sr_evaluation]')
    assert page.locator('[name=frontend]').input_value() == 'browser'
    assert page.locator('[name=decoder]').is_disabled()
