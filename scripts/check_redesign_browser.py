"""Verificação visual local, sem alterar dados ou artefatos científicos.

Requer Playwright apenas no ambiente de desenvolvimento e Microsoft Edge.
Executar com Streamlit em localhost:8503. Capturas ficam fora do repositório.
"""
import json
import math
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


OUT = Path(os.environ.get("TEMP", ".")) / "lendingrisk-redesign"
OUT.mkdir(exist_ok=True)


def wait_ready(page):
    page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=60000)
    assert page.locator('[data-testid="stException"]').count() == 0


def capture(page, name, width):
    wait_ready(page)
    if width < 768:
        page.wait_for_function('document.querySelector("[data-testid=stSidebar]").getBoundingClientRect().right <= 0')
    page.screenshot(path=str(OUT / f"{name}-top.png"))
    # Streamlit scrolls the main element, so expand the viewport to capture its whole content.
    height = math.ceil(page.locator('[data-testid="stMainBlockContainer"]').bounding_box()["height"]) + 30
    page.set_viewport_size({"width": width, "height": height})
    if width < 768:
        page.wait_for_function('document.querySelector("[data-testid=stSidebar]").getBoundingClientRect().right <= 0')
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    page.set_viewport_size({"width": width, "height": 1000})


def check_overflow(page):
    result = page.locator('[data-testid="stMain"]').evaluate(
        '(e) => ({scroll: e.scrollWidth, width: e.clientWidth})')
    assert result["scroll"] <= result["width"] + 1, result
    return result


def navigate(page, name, mobile):
    if page.locator('[data-testid="stSidebar"]').bounding_box()['x'] < 0:
        page.get_by_role("button", name="Expand sidebar").click()
    page.get_by_role("button", name=name, exact=True).click()
    page.get_by_role("heading", name=name, exact=True).wait_for(timeout=60000)
    if mobile:
        page.get_by_role("button", name="Collapse sidebar").click()
    wait_ready(page)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    evidence = []
    for width in [1440, 900, 390, 320]:
        page = browser.new_page(viewport={"width": width, "height": 1000}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        page.goto("http://localhost:8503")
        page.get_by_role("button", name="Estimar risco").wait_for(timeout=60000)
        wait_ready(page)
        assert page.locator('.st-key-proposal [data-testid="stNumberInput"] input').count() == 6
        capture(page, f"simulador-{width}", width)
        check_overflow(page)
        page.locator('.st-key-in_dti input').fill("16.7")
        page.locator('.st-key-in_dti input').press("Tab")
        wait_ready(page)
        page.get_by_role("button", name="Estimar risco").click()
        page.locator('.st-key-detailed-verdict').wait_for(timeout=60000)
        wait_ready(page)
        probability = page.locator('.st-key-summary .lr-hero').inner_text()
        assert probability == '10,7%', probability
        proposal = page.locator('.st-key-proposal').bounding_box()
        detail = page.locator('.st-key-detailed-verdict').bounding_box()
        content = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
        assert detail['y'] > proposal['y'] + proposal['height']
        if width == 1440:
            assert detail['width'] > proposal['width'] * 1.5
        capture(page, f"parecer-{width}", width)
        check_overflow(page)
        navigate(page, "Desempenho dos modelos", width < 768)
        expect(page.locator('[data-testid="stImageContainer"] img')).to_have_count(9, timeout=60000)
        for image in page.locator('[data-testid="stImageContainer"] img').all():
            image.wait_for()
            page.wait_for_function('(e)=>e.complete && e.naturalWidth>0', arg=image.element_handle())
            assert image.evaluate('(e)=>e.complete && e.naturalWidth>0')
        capture(page, f"desempenho-{width}", width)
        check_overflow(page)
        navigate(page, "Dados e metodologia", width < 768)
        expect(page.locator('[data-testid="stImageContainer"] img')).to_have_count(1, timeout=60000)
        capture(page, f"metodologia-{width}", width)
        check_overflow(page)
        # Return through sidebar navigation; simulator title differs from its label.
        if page.locator('[data-testid="stSidebar"]').bounding_box()['x'] < 0:
            page.get_by_role("button", name="Expand sidebar").click()
        page.get_by_role("button", name="Simular empréstimo", exact=True).click()
        page.get_by_role("button", name="Estimar risco").wait_for()
        if width < 768:
            page.get_by_role("button", name="Collapse sidebar").click()
        assert page.locator('.st-key-summary .lr-hero').inner_text() == probability
        page.locator('.st-key-in_loan_amnt input').fill("12000")
        page.locator('.st-key-in_loan_amnt input').press("Tab")
        page.get_by_text("Proposta alterada.", exact=False).wait_for()
        assert page.locator('.st-key-detailed-verdict').count() == 0
        assert not errors, errors
        evidence.append({"width": width, "probability": probability, "console_errors": errors,
                         "performance_figures": 9, "audit_figures": 1, "overflow": False})
        page.close()
    (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({"checks": evidence, "output": str(OUT)}, ensure_ascii=False))
    browser.close()
