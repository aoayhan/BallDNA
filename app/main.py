"""Basketball DNA Streamlit application."""

from __future__ import annotations

import sys
from base64 import b64encode
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ball_ai.analytics.play_style import (  # noqa: E402
    SIAMESE_ENSEMBLE_WEIGHT,
    find_style_neighbors,
)
from ball_ai.analytics.siamese_attribution import (  # noqa: E402
    load_siamese_inference_artifacts,
    siamese_pair_attribution,
)


DATA = ROOT / "data" / "historical"
MODEL = ROOT / "models" / "play_style" / "siamese_tabular_v4_inference.npz"
LOGO = ROOT / "app" / "assets" / "balldna-logo.png"
LOGO_DATA = b64encode(LOGO.read_bytes()).decode("ascii")
IVERSON_DATA = b64encode((ROOT / "app" / "assets" / "allen_iverson_cc_by_sa.jpg").read_bytes()).decode("ascii")
KOFI_DATA = b64encode((ROOT / "app" / "assets" / "kofi-button.png").read_bytes()).decode("ascii")
LEGAL = ROOT / "LEGAL.md"


st.set_page_config(page_title="Basketball DNA · Player similarity", page_icon="◉", layout="wide")

st.markdown(
    ("""
    <style>
    :root {
        --ink: #f5f7fb;
        --muted: #8993a7;
        --panel: rgba(17, 22, 31, .82);
        --line: rgba(255, 255, 255, .09);
        --orange: #ff6b35;
        --lime: #c7f36b;
        --violet: #9e8cff;
    }
    html {color-scheme: dark;}
    .bd-skip {
        position: fixed; top: .5rem; left: .5rem; z-index: 9999; padding: .65rem .9rem;
        color: #090b0f !important; background: var(--lime); border-radius: 8px;
        font-weight: 800; transform: translateY(-160%); transition: transform .12s ease;
    }
    .bd-skip:focus {transform: translateY(0);}
    :focus-visible {outline: 2px solid var(--orange); outline-offset: 3px;}
    ::selection {color: #090b0f; background: var(--lime);}
    header[data-testid="stHeader"], [data-testid="stSidebar"] {display: none;}
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 78% -10%, rgba(158,140,255,.16), transparent 32rem),
            radial-gradient(circle at 2% 42%, rgba(255,107,53,.12), transparent 28rem),
            #080b10;
        color: var(--ink);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1320px;
        padding: .2rem 2.2rem 5rem;
    }
    .st-key-mock_nav {margin-bottom: 1rem; border-bottom: 1px solid var(--line);}
    .st-key-mock_nav [data-testid="stHorizontalBlock"] {align-items: center; min-height: 50px;}
    .st-key-mock_brand {width: 132px;}
    .st-key-mock_brand div[data-testid="stButton"] button {
        width: 132px !important; height: 48px !important; min-height: 48px !important; padding: 0 !important;
        color: transparent !important; font-size: 0 !important; cursor: pointer;
        background: transparent url("data:image/png;base64,LOGO_IMAGE") left center / 132px auto no-repeat !important;
    }
    .st-key-mock_brand:has(button:focus-visible) {outline: 2px solid var(--orange); outline-offset: 5px; border-radius: 6px;}
    .st-key-mock_nav div[data-testid="stButton"] button {
        justify-content: center; padding: .35rem .15rem; color: var(--muted);
        background: transparent; border: 0; border-radius: 0; font-size: .82rem;
        min-height: 2rem; transition: color .18s ease;
    }
    .st-key-mock_nav div[data-testid="stButton"] button:hover {color: var(--ink); background: transparent;}
    .st-key-mock_nav div[data-testid="stButton"] button:focus-visible {outline: 2px solid var(--orange); outline-offset: 4px;}
    .bd-kofi {
        display: inline-flex; justify-content: center; align-items: center; min-height: 36px;
        border-radius: 8px; text-decoration: none !important;
    }
    .bd-kofi img {display: block; width: auto; height: 36px; border: 0;}
    .bd-kofi:hover {filter: brightness(1.08);}
    .bd-kofi:focus-visible {outline: 2px solid var(--orange); outline-offset: 4px;}
    .bd-title {
        max-width: 780px; margin: 0; color: var(--ink); font-size: clamp(2.35rem, 4.8vw, 4.65rem);
        letter-spacing: -.04em; line-height: .92; text-wrap: balance;
    }
    .bd-intro {
        display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, .8fr);
        align-items: center; gap: 2rem; margin-bottom: 1.2rem;
    }
    .bd-intro p {max-width: 48ch; color: var(--muted); font-size: .92rem; line-height: 1.5; margin: 0;}
    .bd-chip-row {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .65rem;}
    .bd-chip {
        border: 1px solid var(--line); border-radius: 999px; padding: .38rem .68rem;
        color: #b7c0d0; background: rgba(255,255,255,.025); font-size: .72rem;
    }
    .st-key-home_hero {
        min-height: 405px; display: flex; justify-content: center; overflow: hidden; position: relative;
        margin: 1rem 0 3rem; padding: clamp(2rem, 5vw, 4rem); border: 1px solid var(--line);
        border-radius: 22px; background: #0b0e14;
        box-shadow: 0 34px 100px rgba(0,0,0,.3);
    }
    .st-key-home_hero::before {
        content: ""; position: absolute; inset: 0; z-index: 0; opacity: .48; filter: grayscale(1);
        background: transparent url("data:image/jpeg;base64,IVERSON_IMAGE") 90% 40% / 52% auto no-repeat;
    }
    .st-key-home_hero::after {
        content: ""; position: absolute; inset: 0; z-index: 0;
        background: linear-gradient(90deg, #0b0e14 0 45%, rgba(11,14,20,.92) 58%, rgba(11,14,20,.18) 100%);
    }
    .bd-home-copy {max-width: 690px; position: relative; z-index: 1;}
    .bd-home-eyebrow, .bd-kicker {
        color: var(--orange); font: 750 .66rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;
        letter-spacing: .14em; text-transform: uppercase;
    }
    .bd-home-title {
        max-width: 690px; margin: .85rem 0 1.15rem; color: var(--ink);
        font-size: clamp(2.65rem, 5.6vw, 5.15rem); line-height: .94; letter-spacing: -.055em;
        text-wrap: balance;
    }
    .bd-home-lede {max-width: 61ch; margin: 0; color: #b1bac9; font-size: 1rem; line-height: 1.65;}
    .bd-home-joke {margin: .9rem 0 0; color: #778195; font-size: .78rem;}
    .st-key-home_actions {max-width: 420px; margin-top: 1rem; position: relative; z-index: 1;}
    .st-key-home_actions [data-testid="stHorizontalBlock"] {
        display: grid; grid-template-columns: 1fr 1.08fr; gap: .65rem;
    }
    .st-key-home_actions [data-testid="stHorizontalBlock"] > div {
        width: auto !important; min-width: 0 !important; flex: none !important;
    }
    .st-key-home_discover button, .st-key-home_compare button {
        min-height: 44px; padding: .65rem 1rem; border-radius: 10px;
        font-size: .78rem; font-weight: 800;
    }
    .st-key-home_discover button {color: #090b0f; background: var(--orange); border-color: var(--orange);}
    .st-key-home_discover button:hover {color: #090b0f; background: #ff8052; border-color: #ff8052;}
    .st-key-home_compare button {color: var(--ink); background: rgba(8,11,16,.68); border-color: var(--line);}
    .st-key-home_compare button:hover {color: var(--ink); border-color: rgba(255,255,255,.22); background: rgba(255,255,255,.06);}
    .st-key-home_actions button:focus-visible {outline: 2px solid var(--lime); outline-offset: 4px;}
    .bd-home-section {margin: 0 0 3.2rem;}
    .bd-home-section-head {display: grid; grid-template-columns: .75fr 1.25fr; gap: 2rem; align-items: end; margin-bottom: 1.35rem;}
    .bd-home-section h2 {margin: .55rem 0 0; font-size: clamp(1.7rem, 3vw, 2.65rem); letter-spacing: -.045em; line-height: 1;}
    .bd-home-section-head p {max-width: 62ch; margin: 0; color: var(--muted); font-size: .86rem; line-height: 1.6;}
    .bd-pipeline {display: grid; grid-template-columns: repeat(4, 1fr); border-block: 1px solid var(--line);}
    .bd-pipeline-step {padding: 1.3rem 1.3rem 1.45rem 0;}
    .bd-pipeline-step + .bd-pipeline-step {padding-left: 1.3rem; border-left: 1px solid var(--line);}
    .bd-pipeline-step span {color: var(--orange); font: 750 .64rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;}
    .bd-pipeline-step h3 {margin: .65rem 0 .45rem; font-size: .93rem; letter-spacing: -.02em;}
    .bd-pipeline-step p {margin: 0; color: var(--muted); font-size: .74rem; line-height: 1.55;}
    .bd-ml-panel {
        display: grid; grid-template-columns: 1.15fr .85fr; gap: 2.5rem; padding: clamp(1.5rem, 3vw, 2.4rem);
        border: 1px solid var(--line); border-radius: 18px; background: linear-gradient(125deg, rgba(255,255,255,.045), rgba(255,255,255,.015));
    }
    .bd-ml-copy h2 {margin: .65rem 0 .9rem; font-size: clamp(1.7rem, 3vw, 2.55rem); letter-spacing: -.045em;}
    .bd-ml-copy p {max-width: 64ch; margin: 0 0 .75rem; color: #9ca6b8; font-size: .82rem; line-height: 1.65;}
    .bd-ml-copy strong {color: var(--ink);}
    .bd-metrics {display: grid; grid-template-columns: repeat(3, 1fr); align-self: center; border-block: 1px solid var(--line);}
    .bd-metric {padding: 1.15rem .8rem; text-align: center;}
    .bd-metric + .bd-metric {border-left: 1px solid var(--line);}
    .bd-metric b {display: block; color: var(--lime); font-size: clamp(1.35rem, 2.8vw, 2rem); letter-spacing: -.04em; font-variant-numeric: tabular-nums;}
    .bd-metric span {display: block; margin-top: .25rem; color: #758095; font-size: .62rem; letter-spacing: .08em; text-transform: uppercase;}
    .bd-metric-note {grid-column: 1 / -1; margin: .9rem 0 0; color: #707b8f; font-size: .69rem; line-height: 1.5; text-align: center;}
    .bd-home-limit {margin: -1.8rem 0 3rem; color: #697488; font-size: .7rem; line-height: 1.5;}
    .st-key-mock_control_shell {
        border: 1px solid var(--line); border-radius: 20px; padding: .7rem 1rem .2rem;
        background: linear-gradient(125deg, rgba(255,255,255,.055), rgba(255,255,255,.018));
        box-shadow: 0 24px 70px rgba(0,0,0,.22); margin-bottom: 1.35rem;
    }
    .bd-control-label {
        color: #687388; font: 700 .65rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: -.2rem;
    }
    div[data-testid="stSelectbox"] label p {
        color: #9aa5b8 !important; font-size: .72rem !important; font-weight: 650 !important;
        letter-spacing: .02em;
    }
    div[data-testid="stSelectbox"] div[role="group"] {
        min-height: 48px; border-radius: 12px; border-color: rgba(255,255,255,.11);
        background: rgba(5,8,13,.7); transition: border-color .18s ease, background .18s ease;
    }
    div[data-testid="stSelectbox"] div[role="group"]:hover {border-color: rgba(255,107,53,.62); background: rgba(12,16,23,.94);}
    div[data-testid="stSelectbox"] div[role="group"]:focus-within {border-color: var(--orange); box-shadow: 0 0 0 3px rgba(255,107,53,.14);}
    div[data-testid="stSelectbox"] input[role="combobox"],
    div[data-testid="stSelectbox"] button {color: var(--ink);}
    div[role="listbox"] {background: #111720; border-color: var(--line);}
    div[role="option"] {color: var(--ink);}
    .bd-field-label {
        color: #9aa5b8; font-size: .72rem; font-weight: 650; letter-spacing: .02em;
        margin: 0 0 .42rem;
    }
    .bd-match-stage {
        position: relative; overflow: hidden; display: grid;
        grid-template-columns: minmax(0, 1fr) 230px minmax(0, 1fr); align-items: stretch;
        min-height: 286px; border: 1px solid var(--line); border-radius: 26px;
        background: linear-gradient(135deg, rgba(19,25,35,.96), rgba(11,14,20,.96));
        box-shadow: 0 34px 100px rgba(0,0,0,.28); margin: 1rem 0 1.15rem;
    }
    .bd-match-stage::before {
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background:
            linear-gradient(90deg, transparent 49.9%, rgba(255,255,255,.06) 50%, transparent 50.1%),
            repeating-linear-gradient(90deg, transparent 0 79px, rgba(255,255,255,.025) 80px),
            repeating-linear-gradient(0deg, transparent 0 79px, rgba(255,255,255,.025) 80px);
        mask-image: linear-gradient(to bottom, transparent, #000 45%, transparent);
    }
    .bd-player {
        position: relative; z-index: 1; display: flex; flex-direction: column;
        justify-content: space-between; padding: 2rem;
    }
    .bd-player:last-child {text-align: right; align-items: flex-end;}
    .bd-rank {color: #667084; font: 700 .68rem/1 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .13em; text-transform: uppercase;}
    .bd-avatar {
        position: relative; overflow: hidden; display: grid; place-items: center;
        width: 96px; height: 96px; border-radius: 24px;
        color: #0a0d12; background: var(--orange); font-size: 1.55rem; font-weight: 900;
        letter-spacing: -.04em; box-shadow: inset 0 0 0 1px rgba(255,255,255,.35), 0 14px 40px rgba(255,107,53,.18);
    }
    .bd-player:last-child .bd-avatar {background: var(--lime); box-shadow: inset 0 0 0 1px rgba(255,255,255,.35), 0 14px 40px rgba(199,243,107,.14);}
    .bd-avatar span {position: absolute; inset: 0; display: grid; place-items: center;}
    .bd-avatar img {position: absolute; inset: 0; z-index: 1; width: 100%; height: 100%; object-fit: cover; object-position: center top;}
    .bd-player-name {font-size: clamp(1.35rem, 2.4vw, 2.1rem); font-weight: 800; letter-spacing: -.04em; line-height: 1.02; max-width: 360px;}
    .bd-player-meta {color: var(--muted); font-size: .78rem; margin-top: .5rem;}
    .bd-score {
        position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center;
        justify-content: center; text-align: center;
    }
    .bd-score-ring {
        display: grid; place-items: center; width: 176px; height: 176px; border-radius: 50%;
        background: radial-gradient(circle at center, #0d1118 57%, transparent 58%),
            conic-gradient(var(--orange) 0 var(--score), rgba(255,255,255,.07) var(--score) 100%);
        box-shadow: 0 0 0 1px rgba(255,255,255,.07), 0 22px 60px rgba(0,0,0,.34);
    }
    .bd-score-value {font-size: 2.25rem; font-weight: 850; letter-spacing: -.035em; font-variant-numeric: tabular-nums;}
    .bd-score-label {color: #7f899b; font: 700 .63rem/1.3 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .1em; text-transform: uppercase;}
    .bd-score-tier {margin-top: .85rem; color: var(--lime); font-size: .72rem; font-weight: 750;}
    .bd-impact-pair {
        display: grid; grid-template-columns: 1fr 1fr; gap: .55rem; width: 220px;
        margin-top: .8rem; padding-top: .7rem; border-top: 1px solid var(--line);
    }
    .bd-impact-pair span {display: flex; flex-direction: column; gap: .2rem;}
    .bd-impact-pair small {color: #707b8f; font-size: .57rem; letter-spacing: .07em; text-transform: uppercase;}
    .bd-impact-pair b {color: var(--ink); font-size: .78rem; font-variant-numeric: tabular-nums;}
    .bd-section-head {display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; margin: 2.1rem 0 .85rem;}
    .bd-section-head h2 {font-size: 1.25rem; letter-spacing: -.035em; margin: 0;}
    .bd-section-head span {color: #697488; font: 600 .68rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;}
    .bd-match-grid {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .7rem;}
    .bd-match-card {
        min-width: 0; min-height: 126px; padding: 1rem; border: 1px solid var(--line); border-radius: 15px;
        background: rgba(255,255,255,.027); transition: transform .18s ease, border-color .18s ease, background .18s ease;
    }
    .bd-match-top {display: flex; justify-content: space-between; color: #697488; font: 650 .64rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;}
    .bd-match-name {margin: 1.25rem 0 .25rem; font-weight: 760; letter-spacing: -.025em; overflow-wrap: anywhere;}
    .bd-match-season {color: var(--muted); font-size: .72rem;}
    .bd-match-score {color: var(--ink); font-size: 1.25rem; font-weight: 800; font-variant-numeric: tabular-nums;}
    .bd-impact {color: var(--lime); font-size: .67rem; margin-top: .45rem; font-variant-numeric: tabular-nums;}
    [class*="st-key-match_card_"] {position: relative;}
    [class*="st-key-match_card_"] > div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) {
        position: absolute; inset: 0; z-index: 2;
    }
    [class*="st-key-match_card_"] div[data-testid="stButton"] {width: 100%; height: 100%;}
    [class*="st-key-match_card_"] div[data-testid="stButton"] button {
        width: 100% !important; height: 100%; min-height: 126px; padding: 0; opacity: 0; cursor: pointer;
    }
    [class*="st-key-match_card_"]:has(button:hover) .bd-match-card {
        transform: translateY(-3px); border-color: rgba(255,255,255,.2); background: rgba(255,255,255,.05);
    }
    [class*="st-key-match_card_"]:has(button:focus-visible) .bd-match-card {
        outline: 2px solid var(--orange); outline-offset: 3px;
    }
    .st-key-match_card_selected .bd-match-card {
        border-color: rgba(255,107,53,.65); background: rgba(255,107,53,.09);
    }
    .st-key-match_cards div[data-testid="stButton"] {width: 100% !important;}
    .st-key-match_cards [data-testid="stHorizontalBlock"] {gap: .7rem;}
    .st-key-mock_show_more button {
        color: var(--ink) !important; border: 1px solid var(--line); border-radius: 12px;
        background: rgba(255,255,255,.035);
    }
    .st-key-mock_show_more button:hover {border-color: rgba(255,107,53,.62); background: rgba(255,107,53,.08);}
    .bd-driver-grid {display: grid; grid-template-columns: 1fr 1fr; gap: .8rem;}
    .bd-driver-panel {border: 1px solid var(--line); border-radius: 18px; padding: 1.2rem; background: var(--panel);}
    .bd-driver-panel h3 {margin: 0 0 1rem; font-size: .84rem; letter-spacing: .01em;}
    .bd-driver {display: grid; grid-template-columns: minmax(0, 1fr) 52px; gap: .85rem; align-items: center; margin: .8rem 0;}
    .bd-driver-name {display: flex; justify-content: space-between; gap: 1rem; color: #b8c0cf; font-size: .76rem;}
    .bd-driver-values {display: flex; flex-wrap: wrap; gap: .3rem .7rem; margin-top: .28rem; color: #778296; font-size: .64rem;}
    .bd-driver-values b {color: #b8c0cf; font-weight: 700; font-variant-numeric: tabular-nums;}
    .bd-driver-track {height: 4px; border-radius: 99px; background: rgba(255,255,255,.07); overflow: hidden; margin-top: .38rem;}
    .bd-driver-fill {height: 100%; border-radius: inherit; background: var(--lime);}
    .bd-driver-panel:last-child .bd-driver-fill {background: var(--violet);}
    .bd-driver-effect {color: #7f899b; text-align: right; font: 650 .68rem/1 ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums;}
    div[data-testid="stExpander"] {border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,.018);}
    div[data-testid="stExpander"] summary:focus-visible {outline: 2px solid var(--orange); outline-offset: 3px;}
    div[data-testid="stPopover"] button {
        justify-content: flex-start; min-height: 48px; border-color: var(--line); border-radius: 12px;
        background: rgba(5,8,13,.7); color: var(--ink);
    }
    div[data-testid="stPopover"] button [data-testid="stIconMaterial"] {margin-left: auto;}
    div[data-testid="stPopover"] button:hover {border-color: rgba(255,107,53,.62); color: var(--ink);}
    .bd-table-wrap {overflow-x: auto; border: 1px solid var(--line); border-radius: 16px; background: var(--panel);}
    .bd-table {width: 100%; border-collapse: collapse; min-width: 760px; font-size: .78rem;}
    .bd-table th {color: #727d91; font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; text-align: left;}
    .bd-table th, .bd-table td {padding: .9rem 1rem; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums;}
    .bd-table tbody tr:last-child td {border-bottom: 0;}
    .bd-table tbody tr:hover {background: rgba(255,255,255,.025);}
    .bd-rank-cell {color: var(--orange); font-weight: 800;}
    .bd-score-cell {color: var(--lime); font-weight: 800;}
    .bd-footnote {color: #697488; font-size: .7rem; line-height: 1.55; margin-top: .8rem;}
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {padding-inline: 1rem;}
        .st-key-mock_nav [data-testid="stHorizontalBlock"] {
            display: grid; grid-template-columns: 1.35fr repeat(4, minmax(0, auto)); gap: .3rem;
        }
        .st-key-mock_nav [data-testid="stHorizontalBlock"] > div {
            width: auto !important; min-width: 0 !important; flex: none !important;
        }
        .st-key-match_cards [data-testid="stHorizontalBlock"] {
            display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem;
        }
        .st-key-match_cards [data-testid="stHorizontalBlock"] > div {
            width: auto !important; min-width: 0 !important; flex: none !important;
        }
        .bd-intro {grid-template-columns: 1fr; gap: 1rem;}
        .st-key-home_hero {
            min-height: 470px; justify-content: flex-end; padding: 2rem;
            background-image:
                radial-gradient(circle at 78% 32%, transparent 0 3rem, rgba(255,107,53,.58) 3.05rem 3.2rem, transparent 3.25rem),
                linear-gradient(0deg, #0b0e14 8%, rgba(11,14,20,.91) 52%, rgba(52,25,20,.52) 100%);
        }
        .bd-home-section-head, .bd-ml-panel {grid-template-columns: 1fr; gap: 1rem;}
        .bd-pipeline {grid-template-columns: repeat(2, 1fr);}
        .bd-pipeline-step:nth-child(3) {border-left: 0;}
        .bd-pipeline-step:nth-child(n+3) {border-top: 1px solid var(--line);}
        .bd-match-stage {grid-template-columns: 1fr;}
        .bd-score {padding: 0 0 1.5rem;}
        .bd-player:last-child {text-align: left; align-items: flex-start;}
        .bd-match-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
        .bd-driver-grid {grid-template-columns: 1fr;}
    }
    @media (max-width: 600px) {
        .st-key-mock_brand {width: 112px;}
        .st-key-mock_brand div[data-testid="stButton"] button {
            width: 112px !important; background-size: 112px auto !important;
        }
        .bd-kofi img {height: 32px;}
        .bd-home-title {font-size: 2.65rem;}
        .bd-pipeline, .bd-metrics {grid-template-columns: 1fr;}
        .bd-pipeline-step, .bd-pipeline-step + .bd-pipeline-step {padding: 1.1rem 0; border-left: 0; border-top: 1px solid var(--line);}
        .bd-pipeline-step:first-child {border-top: 0;}
        .bd-metric + .bd-metric {border-left: 0; border-top: 1px solid var(--line);}
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important;}
    }
    @media (forced-colors: active) {
        .bd-match-card, .bd-driver-panel, .bd-match-stage, .st-key-mock_control_shell {border: 1px solid CanvasText;}
    }
    </style>
    """).replace("LOGO_IMAGE", LOGO_DATA).replace("IVERSON_IMAGE", IVERSON_DATA),
    unsafe_allow_html=True,
)

st.markdown('<a class="bd-skip" href="#ball-dna-main">Skip to main content</a>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_player_dna() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only the frozen assets needed by this isolated concept."""

    embeddings = pd.read_parquet(DATA / "player_style_embeddings.parquet")
    features = pd.read_parquet(DATA / "player_style_features.parquet")
    siamese = pd.read_parquet(DATA / "siamese_offensive_embeddings.parquet")
    history = embeddings.loc[
        embeddings["profile"].eq("Broad history")
        & embeddings["method"].eq("Denoising autoencoder")
        & embeddings["eligible"].astype("boolean").fillna(False)
    ].copy()
    impact = pd.read_parquet(
        DATA / "darko_dpm.parquet",
        columns=["player_id", "season", "dpm", "o_dpm", "d_dpm"],
    ).drop_duplicates(["player_id", "season"], keep="last")
    history = history.drop(columns=["dpm", "o_dpm", "d_dpm"], errors="ignore").merge(
        impact,
        on=["player_id", "season"],
        how="left",
        validate="many_to_one",
    )
    return history, features, siamese


@st.cache_data(show_spinner=False)
def load_leaderboard() -> pd.DataFrame:
    return pd.read_parquet(DATA / "style_similarity_leaderboard.parquet")


@st.cache_resource(show_spinner=False)
def load_attribution_model() -> dict:
    return load_siamese_inference_artifacts(MODEL)


def initials(name: str) -> str:
    return "".join(part[0] for part in name.replace("-", " ").split()[:2]).upper()


def headshot(player_id: object) -> str:
    """Return the public NBA CDN headshot for a numeric player ID."""

    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"


def tier(score: float) -> str:
    if score >= .94:
        return "Basketball doppelgänger"
    if score >= .90:
        return "Style twin"
    if score >= .85:
        return "Close style match"
    if score >= .80:
        return "Some overlap in style"
    if score >= .70:
        return "Partial style match"
    if score >= .60:
        return "Some shared tendencies"
    return "Unlike"


def driver_markup(
    frame: pd.DataFrame,
    metric: str,
    reference_name: str,
    match_name: str,
) -> str:
    rows = frame.loc[frame[metric].gt(0)].nlargest(4, metric)
    maximum = max(float(rows[metric].max()), .0001) if not rows.empty else 1
    short_reference = escape(reference_name.split()[0])
    short_match = escape(match_name.split()[0])
    return "".join(
        f"""
        <div class="bd-driver">
          <div><div class="bd-driver-name"><span>{escape(str(row['Feature']))}</span></div>
          <div class="bd-driver-values"><span>{short_reference} <b>{observed(row['Reference'])}</b></span><span>{short_match} <b>{observed(row['Match'])}</b></span></div>
          <div class="bd-driver-track"><div class="bd-driver-fill" style="width:{100 * float(row[metric]) / maximum:.0f}%"></div></div></div>
          <div class="bd-driver-effect">+{100 * float(row[metric]):.2f} pp</div>
        </div>
        """
        for _, row in rows.iterrows()
    ) or '<div class="bd-footnote">No positive local effects for this pair.</div>'


def signed(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(number) else f"{number:+.2f}"


def observed(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "—" if pd.isna(number) else f"{number:.3f}"


def match_stage_markup(
    reference: pd.Series,
    match: pd.Series,
    score: float,
    *,
    reference_label: str,
    match_label: str,
    lens: str,
    impact_label: str | None = None,
    reference_impact: object = None,
    match_impact: object = None,
) -> str:
    reference_name = str(reference["player_name"])
    match_name = str(match["player_name"])
    impact_context = (
        f'<div class="bd-impact-pair">'
        f'<span><small>Reference {escape(impact_label)}</small><b>{signed(reference_impact)}</b></span>'
        f'<span><small>Match {escape(impact_label)}</small><b>{signed(match_impact)}</b></span>'
        f'</div>'
        if impact_label
        else ""
    )
    return f"""
    <section class="bd-match-stage" aria-label="Focused player similarity match">
      <article class="bd-player">
        <div class="bd-rank">{escape(reference_label)} / {escape(str(reference['season']))}</div>
        <div class="bd-avatar" aria-hidden="true"><span>{escape(initials(reference_name))}</span><img src="{headshot(reference['player_id'])}" alt="" onerror="this.style.display='none'"></div>
        <div><div class="bd-player-name">{escape(reference_name)}</div><div class="bd-player-meta">{escape(str(reference['position']))} · {escape(lens)} Player DNA</div></div>
      </article>
      <div class="bd-score">
        <div class="bd-score-ring" style="--score:{score:.1%}">
          <div><div class="bd-score-value">{score:.2%}</div><div class="bd-score-label">Similarity index</div></div>
        </div>
        <div class="bd-score-tier">{escape(tier(score))}</div>
        {impact_context}
      </div>
      <article class="bd-player">
        <div class="bd-rank">{escape(match_label)} / {escape(str(match['season']))}</div>
        <div class="bd-avatar" aria-hidden="true"><span>{escape(initials(match_name))}</span><img src="{headshot(match['player_id'])}" alt="" onerror="this.style.display='none'"></div>
        <div><div class="bd-player-name">{escape(match_name)}</div><div class="bd-player-meta">{escape(str(match['position']))}</div></div>
      </article>
    </section>
    """


def render_model_drivers(
    features: pd.DataFrame,
    reference: pd.Series,
    match: pd.Series,
    score: float,
) -> float:
    try:
        attributed_score, attribution = siamese_pair_attribution(
            features,
            load_attribution_model(),
            int(reference["player_id"]),
            int(match["player_id"]),
            str(reference["season"]),
            str(match["season"]),
        )
        similarity_drivers = driver_markup(
            attribution,
            "Similarity contribution",
            str(reference["player_name"]),
            str(match["player_name"]),
        )
        difference_drivers = driver_markup(
            attribution,
            "Difference penalty",
            str(reference["player_name"]),
            str(match["player_name"]),
        )
    except (FileNotFoundError, ValueError):
        attributed_score = score
        similarity_drivers = difference_drivers = (
            '<div class="bd-footnote">Attribution unavailable for this pair.</div>'
        )
    st.html(
        f"""
        <div class="bd-section-head"><h2>What the model sees</h2><span>LOCAL PERTURBATION ATTRIBUTION</span></div>
        <div class="bd-driver-grid">
          <section class="bd-driver-panel"><h3>Similarity drivers</h3>{similarity_drivers}</section>
          <section class="bd-driver-panel"><h3>Difference drivers</h3>{difference_drivers}</section>
        </div>
        <div class="bd-footnote">Score effects show how the Siamese model changes when one observed input is neutralized or equalized. They explain this pair locally and do not retrain or alter the ranking.</div>
        """
    )
    return attributed_score


def leaderboard_markup(frame: pd.DataFrame) -> str:
    rows = "".join(
        f"""
        <tr>
          <td class="bd-rank-cell">{rank}</td>
          <td>{escape(str(row.player_a_player_name))}<br><span class="bd-match-season">{escape(str(row.player_a_season))}</span></td>
          <td>{escape(str(row.player_b_player_name))}<br><span class="bd-match-season">{escape(str(row.player_b_season))}</span></td>
          <td class="bd-score-cell">{float(row.similarity_score):.2%}</td>
          <td>{escape(tier(float(row.similarity_score)))}</td>
          <td>{signed(row.player_a_o_dpm)} / {signed(row.player_b_o_dpm)}</td>
        </tr>
        """
        for rank, row in enumerate(frame.itertuples(), start=1)
    )
    return f"""
    <div class="bd-table-wrap"><table class="bd-table">
      <thead><tr><th scope="col">Rank</th><th scope="col">Player A</th><th scope="col">Player B</th><th scope="col">Similarity</th><th scope="col">Match tier</th><th scope="col">O-DPM A / B</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    """


view = str(st.query_params.get("view", "home"))
if view not in {"home", "discover", "compare", "leaderboard", "legal"}:
    view = "home"

page_copy = {
    "discover": (
        "Who plays like your favorite player?",
        "Search three decades of NBA player-seasons using learned basketball behavior.",
    ),
    "compare": (
        "Put two styles under the microscope.",
        "Choose any two eligible player-seasons and inspect their learned similarity, model drivers, and separate impact context.",
    ),
    "leaderboard": (
        "The closest pairs in NBA history.",
        "Explore the highest-scoring different-player season pairs produced by the deployed offensive Player DNA model.",
    ),
}
st.markdown(
    f"""
    <style>
    .st-key-nav_{view} button {{color: var(--ink) !important; font-weight: 700;
        text-decoration: underline !important; text-decoration-color: var(--orange) !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="mock_nav"):
    brand_column, discover_column, compare_column, leaderboard_column, legal_column, support_column = st.columns([5.1, .8, .75, 1, .55, 1.2])
    with brand_column:
        with st.container(key="mock_brand"):
            if st.button("Basketball DNA home", key="go_home"):
                st.query_params["view"] = "home"
                st.rerun()
    for column, name, label in (
        (discover_column, "discover", "Discover"),
        (compare_column, "compare", "Compare"),
        (leaderboard_column, "leaderboard", "Leaderboard"),
        (legal_column, "legal", "Legal"),
    ):
        with column:
            with st.container(key=f"nav_{name}"):
                if st.button(label, key=f"go_{name}", use_container_width=True):
                    st.query_params["view"] = name
                    st.rerun()
    support_column.markdown(
        f'<a class="bd-kofi" href="https://ko-fi.com/Q3A127SI46" target="_blank" rel="noopener noreferrer" aria-label="Support Basketball DNA on Ko-fi"><img src="data:image/png;base64,{KOFI_DATA}" alt="Support Basketball DNA on Ko-fi"></a>',
        unsafe_allow_html=True,
    )

st.markdown('<div id="ball-dna-main" tabindex="-1"></div>', unsafe_allow_html=True)

if view == "legal":
    st.markdown(LEGAL.read_text(encoding="utf-8"))
    st.stop()

if view == "home":
    with st.container(key="home_hero"):
        st.markdown(
            """
          <div class="bd-home-copy">
            <div class="bd-home-eyebrow">NBA player similarity, learned from behavior</div>
            <h1 class="bd-home-title">Find out who your favorite player plays like.</h1>
            <p class="bd-home-lede">Basketball DNA learns Player DNA from NBA shot selection, creation, playmaking, and defensive actions, then searches basketball history for the closest style matches.</p>
            <p class="bd-home-joke">AI here means embeddings—not Allen Iverson, although he is in the search pool.</p>
          </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="home_actions"):
            discover_action, compare_action = st.columns([1, 1.08])
            with discover_action:
                with st.container(key="home_discover"):
                    if st.button("Explore Player DNA", key="home_go_discover", use_container_width=True):
                        st.query_params["view"] = "discover"
                        st.rerun()
            with compare_action:
                with st.container(key="home_compare"):
                    if st.button("Compare two players", key="home_go_compare", use_container_width=True):
                        st.query_params["view"] = "compare"
                        st.rerun()

    st.markdown(
        """
        <section class="bd-home-section">
          <div class="bd-home-section-head">
            <div><div class="bd-kicker">How it works</div><h2>From basketball data to Player DNA.</h2></div>
            <p>The product is built around a retrieval problem: represent how a player behaves, find the nearest player-seasons, and show the evidence without confusing style with talent.</p>
          </div>
          <div class="bd-pipeline">
            <article class="bd-pipeline-step"><span>01 / STRUCTURE</span><h3>Describe the game</h3><p>Shot zones, actions, creation, playmaking, and defensive events become comparable season-level features.</p></article>
            <article class="bd-pipeline-step"><span>02 / LEARN</span><h3>Train persistent style</h3><p>A shared Siamese tabular encoder pulls adjacent seasons from the same player together and pushes different players apart.</p></article>
            <article class="bd-pipeline-step"><span>03 / RETRIEVE</span><h3>Search NBA history</h3><p>Normalized 32-dimensional embeddings rank the nearest eligible player-seasons from 1996–97 onward.</p></article>
            <article class="bd-pipeline-step"><span>04 / EXPLAIN</span><h3>Inspect the evidence</h3><p>Local perturbation attribution reveals which observed behaviors pull each match together or push it apart.</p></article>
          </div>
        </section>

        <section class="bd-home-section bd-ml-panel">
          <div class="bd-ml-copy">
            <div class="bd-kicker">The AI / ML work</div>
            <h2>Self-supervised learning, with the shortcuts removed.</h2>
            <p>I trained shared <strong>64 → 32 MLP encoders</strong> with multi-positive contrastive InfoNCE, using adjacent player seasons and corrupted feature views as positive examples. The deployed representation combines stable cross-era features with richer detail when both seasons have reliable coverage.</p>
            <p>Player identity, name, team, position, height, weight, shooting efficiency, and DARKO impact are excluded from the inputs. That forces the model to learn repeatable basketball behavior rather than memorize who the player is or how good he is.</p>
            <p>The pipeline uses Python, pandas, PyTorch, Parquet model assets, NumPy inference, and Streamlit. Impact and efficiency remain visible as separate context; they never inflate the style score.</p>
          </div>
          <div class="bd-metrics" aria-label="Chronological holdout evaluation">
            <div class="bd-metric"><b>0.670</b><span>Holdout MRR</span></div>
            <div class="bd-metric"><b>54.6%</b><span>Previous season at #1</span></div>
            <div class="bd-metric"><b>82.3%</b><span>Previous season in top 5</span></div>
            <p class="bd-metric-note">Across 19 chronological holdouts. Each season queries an earlier-season pool; success means retrieving that player’s adjacent season near the top.</p>
          </div>
        </section>
        <p class="bd-home-limit">Similarity describes basketball tendencies—not equal ability, quality, or a probability. Defensive Player DNA is still under development and should not be used by itself.</p>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

title, description = page_copy[view]
st.markdown(
    f"""
    <div class="bd-intro">
      <h1 class="bd-title">{escape(title)}</h1>
      <div>
        <p>{escape(description)}</p>
        <div class="bd-chip-row"><span class="bd-chip">1996–2026</span><span class="bd-chip">Self-supervised</span><span class="bd-chip">Impact kept separate</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

history, features, siamese = load_player_dna()
players = (
    history.sort_values("season")
    .drop_duplicates("player_id", keep="last")
    [["player_id", "player_name", "position"]]
    .sort_values("player_name")
)
player_names = players["player_name"].tolist()

if view == "leaderboard":
    leaderboard = load_leaderboard().head(20)
    st.html(
        '<div class="bd-section-head"><h2>Top 20 player-season pairs</h2>'
        '<span>SAME-PLAYER PAIRS EXCLUDED · OFFENSE</span></div>'
        + leaderboard_markup(leaderboard)
        + '<div class="bd-footnote">O-DPM is context only and does not affect the ranking. The similarity index measures behavioral closeness, not equal ability or a probability.</div>'
    )
    st.stop()

if view == "compare":
    offensive_history = history.loc[history["lens"].eq("Offensive")]
    first_default = player_names.index("Shai Gilgeous-Alexander")
    with st.container(key="mock_control_shell"):
        st.markdown('<div class="bd-control-label">Choose two player-seasons</div>', unsafe_allow_html=True)
        first_player_column, first_season_column, second_player_column, second_season_column = st.columns([1.45, .65, 1.45, .65])
        with first_player_column:
            first_name = st.selectbox("Player A", player_names, index=first_default, key="mock_compare_a")
        first_id = int(players.loc[players["player_name"].eq(first_name), "player_id"].iloc[0])
        first_seasons = sorted(offensive_history.loc[offensive_history["player_id"].astype(int).eq(first_id), "season"].unique())
        with first_season_column:
            first_season = st.selectbox("Season A", first_seasons, index=len(first_seasons) - 1)

        second_names = [name for name in player_names if name != first_name]
        second_default = second_names.index("Allen Iverson") if "Allen Iverson" in second_names else 0
        with second_player_column:
            second_name = st.selectbox("Player B", second_names, index=second_default, key="mock_compare_b")
        second_id = int(players.loc[players["player_name"].eq(second_name), "player_id"].iloc[0])
        second_seasons = sorted(offensive_history.loc[offensive_history["player_id"].astype(int).eq(second_id), "season"].unique())
        with second_season_column:
            second_season = st.selectbox("Season B", second_seasons, index=len(second_seasons) - 1)

    reference = offensive_history.loc[
        offensive_history["player_id"].astype(int).eq(first_id)
        & offensive_history["season"].eq(first_season)
    ].iloc[0]
    comparison = find_style_neighbors(
        history,
        first_id,
        lens="Offensive",
        method="Denoising autoencoder",
        season=first_season,
        profile="Broad history",
        top_n=1,
        candidate_player_id=second_id,
        candidate_season_start=second_season,
        candidate_season_end=second_season,
        siamese_embeddings=siamese,
        siamese_weight=SIAMESE_ENSEMBLE_WEIGHT,
    ).iloc[0]
    score = float(comparison["similarity_score"])
    st.markdown(
        match_stage_markup(
            reference,
            comparison,
            score,
            reference_label="Player A",
            match_label="Player B",
            lens="Offensive",
        ),
        unsafe_allow_html=True,
    )
    st.html(
        f'<div class="bd-chip-row"><span class="bd-chip">{escape(first_name)} O-DPM {signed(reference["o_dpm"])}</span>'
        f'<span class="bd-chip">{escape(second_name)} O-DPM {signed(comparison["o_dpm"])}</span>'
        '<span class="bd-chip">Impact does not change similarity</span></div>'
    )
    attributed_score = render_model_drivers(features, reference, comparison, score)
    with st.expander("Model and comparison details"):
        st.markdown(
            f"**Broad History · Offensive · {first_season} vs {second_season}**\n\n"
            f"The directly reconstructed Siamese score is **{attributed_score:.2%}**. "
            "Player identity, team, position, height, weight, and impact are excluded from model inputs."
        )
    st.stop()

default_name = "Stephen Curry"
default_index = player_names.index(default_name) if default_name in player_names else 0
control_shell = st.container(key="mock_control_shell")
with control_shell:
    st.markdown('<div class="bd-control-label">Build a query</div>', unsafe_allow_html=True)
    player_column, lens_column, season_column, mode_column, refinement_column = st.columns(
        [1.45, .72, .62, .85, .58]
    )
    with player_column:
        player_name = st.selectbox("Player", player_names, index=default_index, key="mock_player")
    player_id = int(players.loc[players["player_name"].eq(player_name), "player_id"].iloc[0])
    player_lenses = set(
        history.loc[history["player_id"].astype(int).eq(player_id), "lens"].dropna()
    )
    available_lenses = [
        lens_name
        for lens_name in ("Offensive", "Overall", "Defensive")
        if lens_name in player_lenses
    ]
    if st.session_state.get("mock_lens") not in available_lenses:
        st.session_state["mock_lens"] = available_lenses[0]
    with lens_column:
        lens = st.selectbox("Lens", available_lenses, key="mock_lens")
    eligible_history = history.loc[history["lens"].eq(lens)]
    seasons = sorted(
        eligible_history.loc[
            eligible_history["player_id"].astype(int).eq(player_id), "season"
        ].unique()
    )
    previous_season = st.session_state.get("mock_season")
    if previous_season not in seasons:
        st.session_state["mock_season"] = min(
            seasons,
            key=lambda candidate: abs(
                int(candidate.split("-")[0])
                - int(str(previous_season or seasons[-1]).split("-")[0])
            ),
        )
    with season_column:
        season = st.selectbox("Season", seasons, key="mock_season")
    reference = eligible_history.loc[
        eligible_history["player_id"].astype(int).eq(player_id)
        & eligible_history["season"].eq(season)
    ].iloc[0]
    impact_column = {"Offensive": "o_dpm", "Overall": "dpm", "Defensive": "d_dpm"}[lens]
    impact_label = {"Offensive": "O-DPM", "Overall": "DPM", "Defensive": "D-DPM"}[lens]
    reference_impact = pd.to_numeric(
        pd.Series([reference[impact_column]]), errors="coerce"
    ).iloc[0]
    result_sets = ["Style Twin", "Impact-adjusted"] if pd.notna(reference_impact) else ["Style Twin"]
    if st.session_state.get("mock_result_set") not in result_sets:
        st.session_state["mock_result_set"] = "Style Twin"
    with mode_column:
        result_set = st.selectbox("Result set", result_sets, key="mock_result_set")

    candidate_seasons = sorted(eligible_history["season"].dropna().unique())
    candidate_ages = pd.to_numeric(
        eligible_history["age"], errors="coerce"
    ).dropna().astype(int)
    with refinement_column:
        st.markdown('<div class="bd-field-label">Candidate pool</div>', unsafe_allow_html=True)
        with st.popover("Refine pool", use_container_width=True):
            unique_players = st.toggle(
                "Unique players",
                value=True,
                help="Keep only the highest-ranked season for each matching player.",
            )
            position_aware = st.toggle(
                "Position-aware",
                value=False,
                help="Filter by broad G/F/C labels; position never enters the model.",
            )
            season_range = st.select_slider(
                "Candidate seasons",
                options=candidate_seasons,
                value=(candidate_seasons[0], candidate_seasons[-1]),
            )
            age_range = st.slider(
                "Candidate ages",
                min_value=int(candidate_ages.min()),
                max_value=int(candidate_ages.max()),
                value=(int(candidate_ages.min()), int(candidate_ages.max())),
            )
            impact_tolerance = None
            if result_set == "Impact-adjusted":
                impact_tolerance = st.slider(
                    f"Maximum {impact_label} difference",
                    min_value=.5,
                    max_value=5.0,
                    value=1.5,
                    step=.5,
                )
                st.caption(
                    f"DARKO {impact_label} filters the candidate pool. It never increases a style score."
                )

query_signature = (
    player_id,
    season,
    lens,
    result_set,
    unique_players,
    position_aware,
    season_range,
    age_range,
    impact_tolerance,
)
if st.session_state.get("mock_query_signature") != query_signature:
    st.session_state["mock_query_signature"] = query_signature
    st.session_state["mock_focused_index"] = 0
    st.session_state["mock_match_count"] = 5
match_count = min(int(st.session_state.get("mock_match_count", 5)), 10)

if lens == "Defensive":
    st.error(
        "Defensive Player DNA is still under development and should not be trusted by itself. "
        "Verify it with film, tracking data, and impact context."
    )

try:
    neighbors = find_style_neighbors(
        history,
        player_id,
        lens=lens,
        method="Denoising autoencoder",
        season=season,
        profile="Broad history",
        top_n=match_count,
        position_aware=position_aware,
        impact_tolerance=impact_tolerance,
        candidate_season_start=season_range[0],
        candidate_season_end=season_range[1],
        minimum_age=age_range[0],
        maximum_age=age_range[1],
        siamese_embeddings=siamese if lens == "Offensive" else None,
        siamese_weight=SIAMESE_ENSEMBLE_WEIGHT if lens == "Offensive" else 0.0,
        unique_players=unique_players,
    )
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

pool_label = (
    f"{season_range[0]}–{season_range[1]} · ages {age_range[0]}–{age_range[1]} · "
    f"{lens.lower()} · {'unique players' if unique_players else 'all player-seasons'} · "
    f"{'position-aware' if position_aware else 'all positions'}"
)
with control_shell:
    st.caption(
        pool_label
        + (
            f" · within ±{impact_tolerance:.1f} {impact_label}"
            if impact_tolerance is not None
            else ""
        )
    )

focused_index = min(int(st.session_state.get("mock_focused_index", 0)), len(neighbors) - 1)

match = neighbors.iloc[int(focused_index)]
score = float(match["similarity_score"])
st.markdown(
    match_stage_markup(
        reference,
        match,
        score,
        reference_label="Reference",
        match_label=f"Match #{int(focused_index) + 1}",
        lens=lens,
        impact_label=impact_label if result_set == "Impact-adjusted" else None,
        reference_impact=reference[impact_column],
        match_impact=match[impact_column],
    ),
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="bd-section-head"><h2>{'Impact-adjusted matches' if result_set == 'Impact-adjusted' else 'Closest matches'}</h2><span>{'UNIQUE PLAYERS' if unique_players else 'ALL PLAYER-SEASONS'} · {lens.upper()}</span></div>
    """,
    unsafe_allow_html=True,
)
with st.container(key="match_cards"):
    rows = list(neighbors.itertuples())
    for row_start in range(0, len(rows), 5):
        row_slice = rows[row_start:row_start + 5]
        match_columns = st.columns(len(row_slice))
        for offset, (column, row) in enumerate(zip(match_columns, row_slice)):
            index = row_start + offset
            rank = index + 1
            impact_markup = (
                f'<div class="bd-impact">{impact_label} {signed(getattr(row, impact_column))}</div>'
                if result_set == "Impact-adjusted"
                else ""
            )
            card_markup = (
                '<article class="bd-match-card">'
                f'<div class="bd-match-top"><span>#{rank}</span><span class="bd-match-score">{row.similarity_score:.1%}</span></div>'
                f'<div class="bd-match-name">{escape(str(row.player_name))}</div>'
                f'<div class="bd-match-season">{escape(str(row.season))} · {escape(str(row.position))}</div>'
                f'{impact_markup}</article>'
            )
            card_key = "match_card_selected" if index == focused_index else f"match_card_{rank}"
            with column, st.container(key=card_key):
                st.markdown(card_markup, unsafe_allow_html=True)
                if st.button(
                    f"Inspect {row.player_name} {row.season}",
                    key=f"inspect_match_{rank}",
                    use_container_width=True,
                ):
                    st.session_state["mock_focused_index"] = index
                    st.rerun()
if match_count < 10 and st.button("Show 5 more matches", key="mock_show_more"):
    st.session_state["mock_match_count"] = 10
    st.rerun()

if lens == "Offensive":
    attributed_score = render_model_drivers(features, reference, match, score)
else:
    attributed_score = score
    st.html(
        '<div class="bd-section-head"><h2>Model evidence</h2><span>ENCODER RETRIEVAL</span></div>'
        '<div class="bd-driver-panel"><h3>Local attribution</h3>'
        f'<div class="bd-footnote">Feature-level Siamese attribution is currently available for offensive Player DNA only. {escape(lens)} results still use their validated lens-specific encoder.</div></div>'
    )
with st.expander("Model and retrieval details"):
    st.markdown(
        f"**Broad History · {lens} · {candidate_seasons[0]} to {candidate_seasons[-1]}**\n\n"
        + (
            "Coverage-aware Siamese Player DNA uses stable cross-era behavior plus detailed 2007–08+ action coverage. "
            if lens == "Offensive"
            else f"{lens} results use the validated lens-specific encoder. "
        )
        + f"The focused pair's similarity score is **{attributed_score:.2%}**. "
        "Player identity, team, position, height, and weight are excluded from model inputs.\n\n"
        f"**Candidate pool:** {pool_label}. "
        + (
            f"Candidates are filtered to within ±{impact_tolerance:.1f} {impact_label}; impact does not enter Player DNA."
            if impact_tolerance is not None
            else "Impact is shown separately and does not enter Player DNA."
        )
    )
