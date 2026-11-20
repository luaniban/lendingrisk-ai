"""Componentes visuais sem cálculos do modelo nem dependência de rede."""
from __future__ import annotations

from html import escape
from pathlib import Path
from collections.abc import Callable

import streamlit as st


def markup(value: str) -> None:
    st.markdown(" ".join(line.strip() for line in value.splitlines() if line.strip()),
                unsafe_allow_html=True)


def load_styles() -> None:
    st.html((Path(__file__).resolve().parents[1] / "assets" / "styles.css").read_text(encoding="utf-8"))


def icon(name: str = "chart") -> str:
    paths = {
        "chart": '<path d="M4 4v16h16M8 15l4-5 4 3 4-7"/>',
        "bank": '<path d="m3 8 9-5 9 5M4 9h16M5 10v8m5-8v8m4-8v8m5-8v8M3 21h18"/>',
        "card": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 9h18m-5 6h2"/>',
        "people": '<circle cx="9" cy="8" r="3"/><path d="M3 20v-3a6 6 0 0 1 12 0v3m1-15a3 3 0 0 1 0 6m2 3a5 5 0 0 1 3 4v2"/>',
        "book": '<path d="M5 3h14v18H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Zm0 0v14h14M3 19h16"/>',
        "shield": '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Zm-4 9 3 3 5-6"/>',
        "bolt": '<path d="m13 2-9 12h7l-1 8 10-13h-7Z"/>',
        "sliders": '<path d="M3 7h18M3 17h18M8 4v6m8 4v6"/>',
        "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
    }
    return ('<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
            + paths.get(name, paths["chart"]) + '</svg>')


def brand() -> None:
    # O espaço inicial recebe o controle nativo da sidebar, estilizado como menu.
    markup('<div class="lr-brand"><div><strong>LendingRisk <em>AI</em></strong>'
           '<small>Análise preditiva de crédito</small></div></div>')


def navigation(pages: list[str], current: str, on_select: Callable[[str], None]) -> None:
    """Um único padrão de botão para os três destinos, sem radios decorativos."""
    with st.container(key="navigation"):
        markup(f'<p class="lr-sr-only" role="status" aria-live="polite" aria-atomic="true">'
               f'Página atual: {escape(current)}</p>')
        for slug, label in zip(("simulate", "performance", "methodology"), pages):
            st.button(label, key=f"nav-{slug}", width="stretch",
                      type="primary" if label == current else "tertiary",
                      on_click=on_select, args=(label,))


def connection_status(status: dict) -> None:
    online = status["ok"]
    label = "Ollama ativo" if online else "Ollama indisponível"
    kind = "online" if online else "offline"
    markup(f'<div class="lr-connection"><small>LLM local</small>'
           f'<div role="status" aria-live="polite"><span class="lr-connection-dot {kind}" aria-hidden="true"></span>'
           f'<span>{label}</span></div></div>')


def sidebar_credits() -> None:
    markup('<div class="lr-sidebar-credits"><strong>CARCDSI · IFSP Caraguatatuba</strong>'
           '<span>Hyan Ferreira · Luan Ibanhez</span></div>')


def heading(title: str, description: str, section: str) -> None:
    markup(f'<div class="lr-page-heading"><div class="lr-eyebrow">ANÁLISE PREDITIVA / {escape(section)}</div>'
           f'<h1>{escape(title)}</h1><p class="lr-lead">{escape(description)}</p></div>')


def section(title: str, description: str = "", symbol: str = "chart") -> None:
    markup(f'<div class="lr-card-heading"><span class="lr-icon">{icon(symbol)}</span><div>'
           f'<h3>{escape(title)}</h3>'
           + (f'<p>{escape(description)}</p>' if description else '') + '</div></div>')


def indicators(items: list[tuple[str, str, str, str]]) -> None:
    # CSS grid conserves two columns on narrow phones without shrinking native inputs.
    cards = ''.join(f'<div class="lr-kpi"><div class="lr-kpi-label">{escape(label)}'
                    f'<span class="lr-icon">{icon(symbol)}</span></div>'
                    f'<strong>{escape(value)}</strong><small>{escape(note)}</small></div>'
                    for label, value, note, symbol in items)
    markup(f'<div class="lr-kpis" style="--kpi-count:{len(items)}">{cards}</div>')


def academic_notice() -> None:
    markup('<div class="lr-notice"><b>△ &nbsp; Uso exclusivamente acadêmico</b>'
           '<p>Dados históricos dos Estados Unidos. Não utilize previsões ou explicações deste protótipo '
           'para aprovar ou recusar crédito. A probabilidade é uma estimativa, não uma garantia.</p></div>')


def footer() -> None:
    markup('<div class="lr-footer">LendingRisk AI · CARCDSI — IFSP Caraguatatuba · Projeto acadêmico · 2026</div>')
