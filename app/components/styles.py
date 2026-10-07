"""Small visual system for a polished portfolio demo."""

from __future__ import annotations

import streamlit as st


CSS = """
<style>
    .block-container {max-width: 1180px; padding-top: 1rem; padding-bottom: 4rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(15,23,42,.12);}
    [data-testid="stSidebarHeader"] {
        display: flex; align-items: center; justify-content: space-between;
        padding: 1rem 1rem .5rem;
    }
    [data-testid="stSidebarHeader"]::before {
        content: "Basketball DNA"; color: #152238; font-size: 1.05rem; font-weight: 600;
    }
    [data-testid="stSidebarCollapseButton"] {
        position: static; margin-left: auto; visibility: visible !important;
    }
    header[data-testid="stHeader"] {background: transparent; height: 0; min-height: 0;}
    [data-testid="stToolbar"] {
        display: flex !important; width: 0; height: 0; padding: 0; overflow: visible;
    }
    [data-testid="stToolbarActions"], [data-testid="stMainMenu"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu {
        display: none !important;
    }
    [data-testid="stExpandSidebarButton"] {
        position: fixed; top: .75rem; left: .75rem; z-index: 999;
        display: flex !important; width: 2rem; height: 2rem; visibility: visible !important;
    }
    .hero {
        padding: 2.2rem 2.4rem; border-radius: 22px;
        background: linear-gradient(130deg, #eaf2ff 0%, #f8fbff 60%, #fff0e7 100%);
        border: 1px solid rgba(255,107,53,.28); margin-bottom: 1.4rem;
    }
    .eyebrow {color: #ff996e; font-weight: 700; letter-spacing: .12em; font-size: .78rem;}
    .hero h1 {font-size: 3.1rem; line-height: 1.02; margin: .45rem 0 .8rem;}
    .hero p {color: #43536b; max-width: 760px; font-size: 1.08rem; margin: 0;}
    .callout {
        padding: 1rem 1.15rem; border-radius: 14px; background: rgba(255,107,53,.08);
        border: 1px solid rgba(255,107,53,.2); color: #24324a;
    }
    .feature-card {
        min-height: 165px; padding: 1.15rem; border-radius: 16px;
        border: 1px solid rgba(15,23,42,.10); background: #ffffff;
    }
    .feature-card h3 {font-size: 1.05rem; margin: .25rem 0 .5rem;}
    .feature-card p {color: #526078; font-size: .93rem;}
    div[data-testid="stMetric"] {
        background: #ffffff; border: 1px solid rgba(15,23,42,.10);
        padding: .8rem 1rem; border-radius: 14px;
    }
    .small-muted {color: #64748b; font-size: .84rem;}
    .copyable-table {
        width: 100%; overflow-x: auto; margin: .35rem 0 1rem;
        border: 1px solid rgba(15,23,42,.12); border-radius: 10px;
        background: #ffffff;
    }
    .copyable-table table {
        width: max-content; min-width: 100%; border-collapse: collapse;
        color: #152238; font-size: .88rem; white-space: nowrap;
    }
    .copyable-table th {
        background: #eef3fa; font-weight: 700; text-align: left;
        position: sticky; top: 0; z-index: 1;
    }
    .copyable-table th, .copyable-table td {
        padding: .52rem .7rem; border-bottom: 1px solid rgba(15,23,42,.09);
    }
    .copyable-table tbody tr:last-child td {border-bottom: 0;}
    .copyable-table tbody tr:nth-child(even) {background: #f8fafc;}
    .copyable-table td, .copyable-table th {user-select: text;}
</style>
"""

# ponytail: These stable Streamlit test IDs avoid a custom component; replace this
# hook only if a future Streamlit release changes its shell DOM.
SIDEBAR_SCROLL_SCRIPT = """
<script>
(function () {
const host = window.parent;
const doc = host.document;
const previous = host.__hooplensSidebarScroll;
if (previous) doc.removeEventListener('wheel', previous.handler);
const state = previous?.state ?? {autoCollapsed: false};
const handler = (event) => {
    const delta = event.deltaY;
    host.requestAnimationFrame(() => {
        const scroller = doc.querySelector('[data-testid="stMain"]');
        const sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (!scroller || !sidebar) return;
        const hidden = sidebar.getBoundingClientRect().right <= 0;
        if (delta > 8 && scroller.scrollTop > 80 && !hidden && !state.autoCollapsed) {
            doc.querySelector('[data-testid="stSidebarCollapseButton"] button')?.click();
            state.autoCollapsed = true;
        } else if (delta < -8 && scroller.scrollTop <= 1 && state.autoCollapsed) {
            if (hidden) doc.querySelector('[data-testid="stExpandSidebarButton"]')?.click();
            state.autoCollapsed = false;
        }
    });
};
host.__hooplensSidebarScroll = {handler, state};
doc.addEventListener('wheel', handler, {passive: true});
})();
</script>
"""

PLAYER_SELECT_SCRIPT = """
<script>
(function () {
const host = window.parent;
const doc = host.document;
const previous = host.__hooplensPlayerSelect;
if (previous) {
    doc.removeEventListener('focusin', previous.selectCurrent, true);
    doc.removeEventListener('pointerdown', previous.selectCurrent, true);
    doc.removeEventListener('input', previous.resetResults, true);
}
const playerSelect = (element) => {
    const select = element?.closest?.('[data-testid="stSelectbox"]');
    return select?.parentElement?.textContent.includes('Choose ') ? select : null;
};
const selectCurrent = (event) => {
    const select = playerSelect(event.target);
    if (select) host.setTimeout(() => select.querySelector('input')?.select(), 0);
};
const scrollResultsToTop = () => {
    doc.querySelectorAll('[role="listbox"]').forEach((list) => {
        list.scrollTop = 0;
        if (list.parentElement) list.parentElement.scrollTop = 0;
    });
};
const resetResults = (event) => {
    if (!playerSelect(event.target)) return;
    host.requestAnimationFrame(scrollResultsToTop);
    host.setTimeout(scrollResultsToTop, 60);
};
host.__hooplensPlayerSelect = {selectCurrent, resetResults};
doc.addEventListener('pointerdown', selectCurrent, true);
doc.addEventListener('input', resetResults, true);
})();
</script>
"""


def apply_styles() -> None:
    st.html(
        CSS + SIDEBAR_SCROLL_SCRIPT + PLAYER_SELECT_SCRIPT,
        unsafe_allow_javascript=True,
    )
