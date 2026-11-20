"""Verifica controles globais no Edge; Streamlit deve estar em localhost:8503.

Playwright é uma dependência opcional de desenvolvimento. Não altera dados/modelos.
Capturas e evidências ficam em %TEMP%/lendingrisk-sidebar.
"""
import json
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


OUT = Path(os.environ.get("TEMP", ".")) / "lendingrisk-sidebar"
OUT.mkdir(exist_ok=True)
URL = os.environ.get("LENDINGRISK_BROWSER_URL", "http://127.0.0.1:8503")


def settled(page, expanded):
    expression = ('e.getBoundingClientRect().x >= 0' if expanded else
                  'e.getBoundingClientRect().right <= 0')
    page.wait_for_function(f'() => {{let e=document.querySelector("[data-testid=stSidebar]"); return {expression}}}')
    if not expanded:
        page.wait_for_function('document.querySelector("[data-testid=stMain]").getBoundingClientRect().x < 2')


def nav_geometry(page):
    geometry = page.locator('.st-key-navigation button').evaluate_all('''buttons => buttons.map(e => {
        const r=e.getBoundingClientRect(), text=e.querySelector('p').getBoundingClientRect();
        const css=getComputedStyle(e), icon=getComputedStyle(e,'::before');
        return {x:r.x,width:r.width,height:r.height,textX:text.x,padding:css.padding,
                radius:css.borderRadius,iconWidth:icon.width,iconHeight:icon.height,mask:icon.maskImage};
    })''')
    assert len(geometry) == 3, geometry
    for key in ('x', 'width', 'height', 'textX', 'padding', 'radius', 'iconWidth', 'iconHeight'):
        assert len({item[key] for item in geometry}) == 1, (key, geometry)
    assert len({item['mask'] for item in geometry}) == 3
    return geometry


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    results = []
    for width, touch in [(1440, False), (900, False), (390, True), (320, True)]:
        context = browser.new_context(viewport={"width": width, "height": 1000}, has_touch=touch)
        page = context.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
        page.goto(URL)
        page.get_by_role('button', name='Estimar risco').wait_for(timeout=60000)
        assert page.locator('.lr-topbar, .lr-avatar, .lr-brand-icon').count() == 0
        expect(page.locator('[data-testid=stMainMenuButton]')).to_have_count(0)
        assert page.locator('[data-testid=stHeader]').evaluate('(e)=>e.getBoundingClientRect().height') == 0
        heading = page.locator('.lr-page-heading').first.bounding_box()
        assert 0 <= heading['y'] < 60, heading

        expand = page.get_by_role('button', name='Expand sidebar')
        if expand.count():
            expect(expand).to_be_visible()
            expand.focus()
            page.keyboard.press('Enter')
        settled(page, True)
        collapse = page.get_by_role('button', name='Collapse sidebar')
        expect(collapse).to_be_visible()
        assert collapse.evaluate('(e)=>getComputedStyle(e,"::before").maskImage') != 'none'
        # The native glyph is visually suppressed; the native button remains keyboard accessible.
        expect(collapse.locator('[data-testid=stIconMaterial]')).to_be_hidden()
        geometry = nav_geometry(page)
        assert 'Página atual: Simular empréstimo' in page.locator('.st-key-navigation').aria_snapshot()
        brand = page.locator('.lr-brand strong').bounding_box()
        toggle = collapse.bounding_box()
        assert toggle['x'] + toggle['width'] <= brand['x'] + 1, (toggle, brand)
        assert abs((toggle['y'] + toggle['height']/2) - (brand['y'] + brand['height']/2)) < 16
        credits = page.locator('.lr-sidebar-credits')
        assert 'Projeto final' not in credits.inner_text()
        assert credits.bounding_box()['y'] > 850, credits.bounding_box()
        assert page.locator('[data-testid=stRadioOption]').count() == 0

        status = page.locator('.st-key-ollama-status')
        refresh = page.get_by_role('button', name='Verificar conexão', exact=True)
        expect(refresh).to_have_count(1)
        assert page.get_by_role('button', name='Verificar Ollama de novo').count() == 0
        if touch:
            expect(refresh).to_have_css('opacity', '1')
        else:
            page.locator('.lr-brand').hover()
            expect(refresh).to_have_css('opacity', '0')
            before = status.bounding_box()
            status.hover()
            expect(refresh).to_have_css('opacity', '1')
            assert status.bounding_box() == before
            refresh.hover()
            expect(page.get_by_role('tooltip')).to_contain_text('Verificar conexão')
            page.locator('.lr-brand').hover()
            expect(refresh).to_have_css('opacity', '0')
            refresh.focus()
            expect(refresh).to_have_css('opacity', '1')
        # Enter triggers the actual local availability check. Mocks exercise offline in AppTest.
        refresh.focus()
        page.keyboard.press('Enter')
        expect(status.locator('[role=status]')).to_contain_text('Ollama', timeout=60000)
        assert status.locator('.lr-connection-dot.online, .lr-connection-dot.offline').count() == 1
        page.screenshot(path=str(OUT / f'sidebar-expanded-{width}.png'))

        # Active and inactive links retain identical geometry after navigation.
        for label in ['Desempenho dos modelos', 'Dados e metodologia', 'Simular empréstimo']:
            page.get_by_role('button', name=label, exact=True).click()
            page.wait_for_function('''label => {
                const buttons=document.querySelectorAll('.st-key-navigation button[kind=primary]');
                return buttons.length===1 && buttons[0].textContent===label;
            }''', arg=label)
            assert f'Página atual: {label}' in page.locator('.st-key-navigation').aria_snapshot()
            expected_images = {'Desempenho dos modelos': 9, 'Dados e metodologia': 1, 'Simular empréstimo': 0}[label]
            images = page.locator('[data-testid=stImageContainer] img')
            expect(images).to_have_count(expected_images)
            for image in images.all():
                page.wait_for_function('(e)=>e.complete && e.naturalWidth>0', arg=image.element_handle())
            nav_geometry(page)
        collapse.focus()
        page.keyboard.press('Enter')
        settled(page, False)
        expect(page.get_by_role('button', name='Expand sidebar')).to_be_visible()
        assert page.get_by_role('button', name='Collapse sidebar').count() == 0
        main = page.locator('[data-testid=stMain]').bounding_box()
        assert main['x'] < 2, main
        hamburger = page.get_by_role('button', name='Expand sidebar').bounding_box()
        eyebrow = page.locator('.lr-page-heading .lr-eyebrow').first.bounding_box()
        assert hamburger['x'] + hamburger['width'] <= eyebrow['x'] or hamburger['y'] + hamburger['height'] <= eyebrow['y'], (hamburger, eyebrow)
        page.screenshot(path=str(OUT / f'sidebar-collapsed-{width}.png'))
        scroll = page.locator('[data-testid=stMain]').evaluate('(e)=>[e.scrollWidth,e.clientWidth]')
        assert scroll[0] <= scroll[1] + 1, scroll
        assert not errors, errors
        results.append({'width': width, 'touch': touch, 'navigation': 'uniforme', 'hamburger': 'teclado e mouse',
                        'refresh': 'real; hover/foco/toque verificados', 'console_errors': errors})
        context.close()
    browser.close()
    (OUT / 'evidence.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'checks': results, 'captures': str(OUT)}, ensure_ascii=False))
