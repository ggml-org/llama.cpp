import { CHAT_FORM_RESIZE_MS } from '$lib/constants';

export interface CollapsedHeightHoldOptions {
	getCard: () => HTMLElement | undefined;
	/** The editor's box, which sets its minimum and maximum height. */
	getEditorBox: () => HTMLElement | undefined;
	/** The element holding the editor's text. */
	getContent: () => HTMLElement | undefined;
	getValue: () => string;
}

/**
 * Holds the collapsed card's real height in `--chat-form-collapsed-height` from expanding
 * until the collapse has settled, so a quick toggle never measures a card mid-transition.
 */
export function createCollapsedHeightHold(options: CollapsedHeightHoldOptions) {
	let hold: {
		card: number;
		content: number;
		editorMax: number;
		editorMin: number;
		text: string;
	} | null = null;
	let releaseTimer: ReturnType<typeof setTimeout> | undefined;

	const contentHeight = () => options.getContent()?.offsetHeight ?? 0;

	function setHeight(px: number | null) {
		const style = options.getCard()?.style;

		if (px === null) style?.removeProperty('--chat-form-collapsed-height');
		else style?.setProperty('--chat-form-collapsed-height', `${px}px`);
	}

	/** Before the form expands. */
	function start() {
		const card = options.getCard();
		const box = options.getEditorBox();

		clearTimeout(releaseTimer);

		if (hold || !card || !box) return;

		const editor = getComputedStyle(box);

		hold = {
			card: card.getBoundingClientRect().height,
			content: contentHeight(),
			editorMax: parseFloat(editor.maxHeight) || Infinity,
			editorMin: parseFloat(editor.minHeight) || 0,
			text: options.getValue()
		};

		// in place with no transition of its own, so the one the switch starts begins here
		card.style.transition = 'none';
		setHeight(hold.card);
		void getComputedStyle(card).minHeight;
		card.style.transition = '';
	}

	/** Before the form collapses: the height it settles at, held until the transition is over. */
	function settle() {
		if (!hold) return;

		// the held height, unless the text changed meanwhile: then the height the card will
		// have with it, its editor clamped as when collapsed
		const held = hold;
		const editorBox = (content: number) =>
			Math.max(held.editorMin, Math.min(content, held.editorMax));

		setHeight(
			options.getValue() === held.text
				? held.card
				: held.card - editorBox(held.content) + editorBox(contentHeight())
		);

		clearTimeout(releaseTimer);
		releaseTimer = setTimeout(() => {
			hold = null;
			setHeight(null);
		}, CHAT_FORM_RESIZE_MS);
	}

	function destroy() {
		clearTimeout(releaseTimer);
	}

	return { destroy, settle, start };
}
