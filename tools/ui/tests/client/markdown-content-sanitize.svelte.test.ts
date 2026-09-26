// Adversarial checks for MarkdownContent's allowHtml mode: the discover README
// renders HuggingFace model cards, which are third-party content, so every
// payload below must come out neutralized - both right after sanitize and
// after an innerHTML re-parse, which is where mutation XSS would surface.
import MarkdownContent from '$lib/components/app/content/MarkdownContent/MarkdownContent.svelte';
import { SAFE_HTML_CONFIG } from '$lib/components/app/content/MarkdownContent/safe-html-config';
import DOMPurify from 'dompurify';
import { mount, unmount } from 'svelte';
import { describe, expect, it } from 'vitest';

const HANDLER_ATTR = /\son[a-z]+\s*=/i;
const JS_URL = /javascript\s*:/i;

/** Payload execution canary: set by any payload that runs. */
function xssFired(): boolean {
	return (window as { __mdXss?: number }).__mdXss !== undefined;
}

function roundTrip(html: string): string {
	const el = document.createElement('div');

	el.innerHTML = html;
	document.body.appendChild(el);
	const again = el.innerHTML;

	el.remove();

	return again;
}

const PAYLOADS: Record<string, string> = {
	'details ontoggle': '<details open ontoggle="window.__mdXss=1">x</details>',
	'dom clobbering': '<p id="content" name="location"><input name="domain"></p>',
	'form and input': '<form><input autofocus onfocus="window.__mdXss=1"></form>',
	'iframe javascript src': '<iframe src="javascript:alert(1)"></iframe>',
	'iframe srcdoc': '<iframe srcdoc="<script>window.__mdXss=1</script>"></iframe>',
	'img onerror': '<img src=x onerror="window.__mdXss=1">',
	'link case and entity href': '<a href="JaVaScRiPt&colon;alert(1)">x</a>',
	'link data:text/html href':
		'<a href="data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==">x</a>',
	'link javascript href': '<a href="javascript:alert(1)">x</a>',
	'link vbscript href': '<a href="vbscript:msgbox(1)">x</a>',
	'math href': '<math href="javascript:alert(1)"></math>',
	'mathml annotation-xml integration point':
		'<math><annotation-xml encoding="text/html"><img src=x onerror="window.__mdXss=1"></annotation-xml></math>',
	'mathml mtext style mXSS':
		'<math><mtext><table><mglyph><style><img src=x onerror="window.__mdXss=1"></style></mglyph></table></mtext></math>',
	noscript: '<noscript><p title="</noscript><img src=x onerror="window.__mdXss=1>">',
	script: '<script>window.__mdXss=1</script>',
	'srcset javascript': '<img srcset="javascript:alert(1) 1x, x 2x" src=x>',
	'style tag': '<style>@import url(javascript:alert(1));</style>',
	'svg foreignObject smuggling':
		'<svg><foreignObject><img src=x onerror="window.__mdXss=1"></foreignObject></svg>',
	'svg set-attribute animation': '<svg><set attributeName="onmouseover" to="alert(1)"/></svg>',
	'unclosed tag': '<img src="x" onerror="window.__mdXss=1"',
	'uppercase tags': '<IMG SRC=x ONERROR="window.__mdXss=1">',
	'video source onerror': '<video><source onerror="window.__mdXss=1" src=x></video>',
	'xlink href': '<math><mtext xlink:href="javascript:alert(1)"></mtext></math>'
};

describe('SAFE_HTML_CONFIG adversarial battery', () => {
	it('neutralizes every payload after sanitize and after innerHTML re-parse', () => {
		const failures: string[] = [];

		for (const [name, payload] of Object.entries(PAYLOADS)) {
			const clean = DOMPurify.sanitize(payload, SAFE_HTML_CONFIG) as string;
			const again = roundTrip(clean);

			if (HANDLER_ATTR.test(clean) || HANDLER_ATTR.test(again)) {
				failures.push(`${name}: handler attribute survived: ${again}`);
			}

			if (JS_URL.test(clean) || JS_URL.test(again)) {
				failures.push(`${name}: javascript: URL survived: ${again}`);
			}
		}

		expect(failures).toEqual([]);
	});

	it('keeps benign model-card markup intact', () => {
		const clean = DOMPurify.sanitize(
			'<h2 id="title">Title</h2><p>Text with <a href="https://example.com" target="_blank" rel="noopener">a link</a>, <img src="https://example.com/i.png" alt="i" width="100">, <code>code</code>, <table><tr><td colspan="2">cell</td></tr></table>, <math><semantics><annotation encoding="application/x-tex">x^2</annotation></semantics></math></p>',
			SAFE_HTML_CONFIG
		) as string;

		expect(clean).toContain('href="https://example.com"');
		expect(clean).toContain('<img');
		expect(clean).toContain('colspan="2"');
		expect(clean).toContain('<annotation');
	});
});

describe('MarkdownContent end to end', () => {
	it('renders an allowHtml payload README without executing it', async () => {
		const evil = [
			'# Evil card',
			'',
			'<img src=x onerror="window.__mdXss=1">',
			'<a href="javascript:alert(1)">x</a>',
			'<script>window.__mdXss=1</script>',
			'<iframe srcdoc="<script>window.__mdXss=1</script>"></iframe>',
			'<math><annotation-xml encoding="text/html"><img src=x onerror="window.__mdXss=1"></annotation-xml></math>',
			'',
			'normal **markdown** continues'
		].join('\n');
		const target = document.createElement('div');

		document.body.appendChild(target);
		mount(MarkdownContent, { props: { allowHtml: true, content: evil }, target });

		for (let i = 0; i < 100 && !target.textContent?.includes('normal'); i++) {
			await new Promise((r) => setTimeout(r, 50));
		}
		await new Promise((r) => setTimeout(r, 200));

		expect(target.textContent).toContain('normal');
		expect(xssFired()).toBe(false);
		expect(target.querySelector('script')).toBeNull();
		expect(target.querySelector('iframe')).toBeNull();

		target.remove();
	});

	it('escapes raw HTML to literal text in the default mode', async () => {
		const target = document.createElement('div');

		document.body.appendChild(target);
		const component = mount(MarkdownContent, {
			props: {
				content: '<script>window.__mdXss=1</script><img src=x onerror="window.__mdXss=1">'
			},
			target
		});

		for (let i = 0; i < 100 && !target.textContent?.includes('onerror'); i++) {
			await new Promise((r) => setTimeout(r, 50));
		}
		await new Promise((r) => setTimeout(r, 200));

		expect(xssFired()).toBe(false);
		expect(target.querySelector('img[src="x"]')).toBeNull();
		expect(target.textContent).toContain('onerror');

		target.remove();

		if (component) unmount(component);
	});
});
