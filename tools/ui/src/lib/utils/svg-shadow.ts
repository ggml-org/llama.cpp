/**
 * Mounts svg markup inside an open shadow root on the host element.
 * The shadow boundary scopes the svg <style> and its animations to the host,
 * so model authored css can not reach the surrounding page. The caller passes
 * markup that is already sanitized, this only isolates and sizes it.
 */
export function mountSvgShadow(host: HTMLElement, markup: string, style: string): void {
	const root = host.shadowRoot ?? host.attachShadow({ mode: 'open' });
	root.innerHTML = markup ? `<style>${style}</style>${markup}` : '';
}
