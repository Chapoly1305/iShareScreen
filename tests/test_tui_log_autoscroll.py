"""Regression: the session-screen log holds its scroll position when the user
scrolls up, instead of snapping back to the bottom on every new line (stock
RichLog does the latter — Textual #6311). Driven by the scroll reactive in
``_AutoPauseRichLog.watch_scroll_y``."""
import asyncio

from textual.app import App, ComposeResult

from isharescreen.tui.session_screen import LogPanel


class _LogApp(App):
    def compose(self) -> ComposeResult:
        yield LogPanel(id="log")


async def _exercise() -> None:
    app = _LogApp()
    async with app.run_test(size=(80, 24)) as pilot:
        panel = app.query_one("#log", LogPanel)
        view = panel.query_one("#log-view")

        # Fill past one screen — while pinned to the bottom it should follow.
        for i in range(200):
            panel.append(f"INFO line {i}")
        await pilot.pause()
        assert view.auto_scroll, "should follow the tail while at the bottom"
        bottom_y = view.scroll_offset.y
        assert bottom_y > 0, "content should have scrolled"

        # User scrolls up: new lines must NOT yank the viewport back down.
        view.scroll_to(y=0, animate=False)
        await pilot.pause()
        for i in range(200, 215):
            panel.append(f"INFO line {i}")
        await pilot.pause()
        assert not view.auto_scroll, "must pause auto-scroll while scrolled up"
        assert view.scroll_offset.y < bottom_y, "must hold scroll position"
        assert view._pending_while_paused >= 15, "should count lines arriving while paused"

        # Scrolling back to the bottom resumes following.
        view.scroll_end(animate=False)
        await pilot.pause()
        panel.append("INFO back at bottom")
        await pilot.pause()
        assert view.auto_scroll, "must resume following once back at the bottom"


def test_log_auto_pauses_on_scroll_up() -> None:
    asyncio.run(_exercise())
