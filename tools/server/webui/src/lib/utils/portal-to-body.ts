export function portalToBody(node: HTMLElement) {
	if (typeof document === 'undefined') {
		return;
	}

	const target = document.body;
	if (!target) {
		return;
	}

	target.appendChild(node);

	return {
		destroy() {
			if (node.parentNode === target) {
				target.removeChild(node);
			}
		}
	};
}
