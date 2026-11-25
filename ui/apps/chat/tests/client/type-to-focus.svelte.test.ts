import { shouldTypeToFocus } from '#lib/utils/type-to-focus.ts';
import { afterEach, describe, expect, it } from 'vitest';

function key(key: string, init: KeyboardEventInit = {}) {
	return new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key, ...init });
}

function mount(html: string) {
	const host = document.createElement('div');

	host.innerHTML = html;
	document.body.appendChild(host);

	return host;
}

describe('shouldTypeToFocus', () => {
	afterEach(() => {
		(document.activeElement as HTMLElement | null)?.blur();
		document.body.innerHTML = '';
	});

	it('takes a printable key when nothing has focus', () => {
		expect(shouldTypeToFocus(key('a'))).toBe(true);
		expect(shouldTypeToFocus(key('A', { shiftKey: true }))).toBe(true);
		expect(shouldTypeToFocus(key('?'))).toBe(true);
	});

	it('takes a printable key while a button has focus', () => {
		mount('<button>Copy</button>').querySelector('button')!.focus();

		expect(shouldTypeToFocus(key('h'))).toBe(true);
	});

	it('leaves named keys, whitespace and shortcuts alone', () => {
		for (const name of ['Enter', 'Escape', 'Tab', 'ArrowDown', 'F5', ' ']) {
			expect(shouldTypeToFocus(key(name))).toBe(false);
		}

		expect(shouldTypeToFocus(key('k', { metaKey: true }))).toBe(false);
		expect(shouldTypeToFocus(key('k', { ctrlKey: true }))).toBe(false);
		expect(shouldTypeToFocus(key('k', { altKey: true }))).toBe(false);
	});

	it('leaves keys another handler already took', () => {
		const event = key('a');

		event.preventDefault();

		expect(shouldTypeToFocus(event)).toBe(false);
	});

	it('leaves typing in a field or editor where it is', () => {
		const host = mount('<input /><textarea></textarea><div contenteditable="true">x</div>');

		for (const el of host.querySelectorAll<HTMLElement>('input, textarea, [contenteditable]')) {
			el.focus();
			expect(shouldTypeToFocus(key('a'))).toBe(false);
		}
	});

	it('leaves keys inside dialogs and menus', () => {
		const host = mount(
			'<div role="dialog"><button>OK</button></div><div role="menu"><button role="menuitem">Rename</button></div>'
		);

		for (const button of host.querySelectorAll('button')) {
			button.focus();
			expect(shouldTypeToFocus(key('r'))).toBe(false);
		}
	});
});
