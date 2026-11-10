import { useKeyboardInset } from './use-keyboard-inset.svelte';
import {
	SHEET_AT_REST,
	type SheetFrame,
	sheetFrameAt,
	type SheetRects,
	shiftSheetRects,
	shouldExpandChatForm
} from '$lib/components/app/chat/ChatForm/chat-form-expand';
import {
	CHAT_FORM_EASE_OUT,
	CHAT_FORM_RESIZE_MS,
	CHAT_FORM_SHEET_MS,
	UI_DATA_ATTRS
} from '$lib/constants';
import { deviceStore, uiStore } from '$lib/stores';
import { flushSync, onDestroy, tick, untrack } from 'svelte';

export interface UseChatFormExpansionOptions {
	/** Whether this form expands at all. */
	isEnabled: () => boolean;
	getValue: () => string;
	/** The form's card, and the box around its actions. */
	getCard: () => HTMLElement | undefined;
	getActions: () => HTMLElement | undefined;
	focusInput: () => void;
}

/**
 * The chat form's expanded editor: entered once the message asks for it (see
 * shouldExpandChatForm) or by hand, kept while the message has text, and left once it
 * is cleared or sent. Collapsing by hand holds until then too.
 *
 * On a phone the expanded form is a full-screen sheet: expanding switches to the
 * sheet's layout at once and eases it from the collapsed card's place, collapsing
 * slides it back and only then switches the layout (see sheetFrameAt).
 */
export function useChatFormExpansion(options: UseChatFormExpansionOptions) {
	let isExpanded = $state(false);
	let isCollapsedByUser = false;
	let hadText = false;
	let wasExpanded = false;
	let isSheetMoving = $state(false);
	// WebKit builds the card's backdrop-filter layer with the animation's last clip when
	// the blur comes back as the card lands: it waits for the move to be over
	let isBlurHeld = $state(false);
	let collapsedRects: SheetRects | null = null;
	// iOS Safari resizes its viewport while the sheet is up: the collapsed card's place
	// is held relative to the sheet's bottom edge
	let sheetBottomAtOpen = 0;

	$effect(() => {
		if (!options.isEnabled()) return;

		const text = options.getValue() ?? '';

		untrack(() => {
			if (!text.trim()) {
				// cleared or sent: straight back to the small form, without the sheet's slide
				if (hadText) {
					isCollapsedByUser = false;

					if (isSheetMoving) clearSheetAnimations();

					isExpanded = false;
				}

				hadText = false;

				return;
			}

			hadText = true;

			if (!isExpanded && !isCollapsedByUser) isExpanded = shouldExpandChatForm(text);
		});
	});

	$effect(() => {
		if (options.isEnabled()) uiStore.isComposerExpanded = isExpanded;
	});

	// the phone sheet fills the screen, so its actions would sit under an iOS keyboard
	useKeyboardInset(() => isExpanded && deviceStore.isMobile);

	// the page around the phone sheet takes its fill (see app.css) once it has settled:
	// repainting the whole page under it mid-move costs frames
	$effect(() => {
		if (!options.isEnabled() || !isExpanded || isSheetMoving) return;

		document.documentElement.setAttribute(UI_DATA_ATTRS.CHAT_FORM_SHEET, '');

		return () => document.documentElement.removeAttribute(UI_DATA_ATTRS.CHAT_FORM_SHEET);
	});

	function animatesSheet(): boolean {
		return deviceStore.isMobile && !matchMedia('(prefers-reduced-motion: reduce)').matches;
	}

	function sheetBottom(): number {
		return options.getCard()?.closest('form')?.getBoundingClientRect().bottom ?? 0;
	}

	function measure(): SheetRects | null {
		const card = options.getCard();
		const actions = options.getActions();

		if (!card || !actions) return null;

		const { height, left, right, top } = card.getBoundingClientRect();

		return {
			actionsTop: actions.getBoundingClientRect().top,
			card: { height, left, right, top },
			padTop: parseFloat(getComputedStyle(card).paddingTop) || 0
		};
	}

	/**
	 * Plays the frames on the card and its actions. Nothing holds the last frame, as
	 * WebKit can keep drawing a held frame's clip after the layout changed: a move that
	 * ends where the layout does not (collapsing) switches the layout in `onEnd`, as the
	 * animation ends and before that frame is painted.
	 */
	function animateSheet(
		frames: SheetFrame[],
		duration = CHAT_FORM_SHEET_MS,
		onEnd?: () => void
	): Promise<unknown> {
		const keyframeOptions = { duration, easing: CHAT_FORM_EASE_OUT };
		const animations = [
			options.getCard()!.animate(
				frames.map((frame) => frame.card),
				keyframeOptions
			),
			options.getActions()!.animate(
				frames.map((frame) => frame.actions),
				keyframeOptions
			)
		];

		if (onEnd) animations[0].addEventListener('finish', onEnd, { once: true });

		return Promise.all(animations.map((animation) => animation.finished));
	}

	function clearSheetAnimations() {
		for (const el of [options.getCard(), options.getActions()]) {
			el?.getAnimations()
				.filter((animation) => !(animation instanceof CSSTransition))
				.forEach((animation) => animation.cancel());
		}
	}

	async function expandSheet(from: SheetRects) {
		isSheetMoving = true;
		// the page lays the form out flush with the screen in the same update: measure
		// once all of it has landed, before a paint
		await tick();
		sheetBottomAtOpen = sheetBottom();

		try {
			await animateSheet([sheetFrameAt(measure()!, from), SHEET_AT_REST]);
		} catch {
			// cancelled: the form went away or collapsed
		}

		clearSheetAnimations();
		isSheetMoving = false;
	}

	async function collapseSheet() {
		const sheet = measure();

		if (!collapsedRects || !sheet) {
			isExpanded = false;

			return;
		}

		const target = shiftSheetRects(collapsedRects, sheetBottom() - sheetBottomAtOpen);

		isSheetMoving = true;
		isBlurHeld = true;

		try {
			await animateSheet([SHEET_AT_REST, sheetFrameAt(sheet, target)], CHAT_FORM_SHEET_MS, () => {
				flushSync(() => (isExpanded = false));
			});
		} catch {
			// cancelled: settle at once
		}

		isExpanded = false;
		await tick();

		// the text may have changed while expanded, and with it the collapsed height:
		// close the small remaining gap the same way
		const settled = measure();

		if (settled && Math.abs(settled.card.top - target.card.top) > 1) {
			const dy = target.card.top - settled.card.top;

			await animateSheet(
				[
					{ actions: {}, card: { top: `${dy}px` } },
					{ actions: {}, card: { top: '0px' } }
				],
				CHAT_FORM_SHEET_MS / 2
			).catch(() => {});
			clearSheetAnimations();
		}

		isSheetMoving = false;

		await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
		isBlurHeld = false;
	}

	// The expanded size is built on the collapsed card's height, and the height
	// transition starts from it. The real height is held in a variable on the card from
	// expanding until the collapse has settled, so a quick toggle back and forth never
	// measures a card that is halfway through a transition.
	let collapsedHold: {
		card: number;
		content: number;
		editorMax: number;
		editorMin: number;
		text: string;
	} | null = null;
	let collapsedReleaseTimer: ReturnType<typeof setTimeout> | undefined;

	function editorContentHeight(): number {
		return options.getCard()?.querySelector<HTMLElement>('.cm-content')?.offsetHeight ?? 0;
	}

	function setCollapsedHeight(px: number | null) {
		const style = options.getCard()?.style;

		if (px === null) style?.removeProperty('--chat-form-collapsed-height');
		else style?.setProperty('--chat-form-collapsed-height', `${px}px`);
	}

	function holdCollapsedHeight() {
		const card = options.getCard();

		clearTimeout(collapsedReleaseTimer);

		if (collapsedHold || !card) return;

		const editor = getComputedStyle(card.querySelector<HTMLElement>('.chat-form-input')!);

		collapsedHold = {
			card: card.getBoundingClientRect().height,
			content: editorContentHeight(),
			editorMax: parseFloat(editor.maxHeight) || Infinity,
			editorMin: parseFloat(editor.minHeight) || 0,
			text: options.getValue() ?? ''
		};

		// in place before the switch, with no transition of its own, so the height
		// transition the switch starts begins at the card's real height
		card.style.transition = 'none';
		setCollapsedHeight(collapsedHold.card);
		void getComputedStyle(card).minHeight;
		card.style.transition = '';
	}

	function settleCollapsedHeight() {
		if (!collapsedHold) return;

		// Back to the held height, unless the text changed while expanded: then to the
		// height the card will have with it, its editor clamped as when collapsed. The
		// editor is only measured then, as mid-move its height is not the text's
		const hold = collapsedHold;
		const editorBox = (content: number) =>
			Math.max(hold.editorMin, Math.min(content, hold.editorMax));

		setCollapsedHeight(
			(options.getValue() ?? '') === hold.text
				? hold.card
				: hold.card - editorBox(hold.content) + editorBox(editorContentHeight())
		);

		clearTimeout(collapsedReleaseTimer);
		collapsedReleaseTimer = setTimeout(() => {
			collapsedHold = null;
			setCollapsedHeight(null);
		}, CHAT_FORM_RESIZE_MS);
	}

	// before the DOM switches: where the collapsed card is, and how tall
	$effect.pre(() => {
		const expanded = isExpanded;

		untrack(() => {
			if (expanded === wasExpanded) return;

			if (!expanded) {
				settleCollapsedHeight();

				return;
			}

			if (!isSheetMoving && animatesSheet()) collapsedRects = measure();

			holdCollapsedHeight();
		});
	});

	// after it switched: ease the sheet in from there
	$effect(() => {
		const expanded = isExpanded;

		untrack(() => {
			if (expanded && !wasExpanded && collapsedRects && !isSheetMoving) {
				void expandSheet(collapsedRects);
			}

			if (!expanded) collapsedRects = null;

			wasExpanded = expanded;
		});
	});

	onDestroy(() => {
		clearTimeout(collapsedReleaseTimer);

		if (options.isEnabled()) uiStore.isComposerExpanded = false;
	});

	function toggle() {
		if (isSheetMoving) return;

		// collapsing an empty form leaves the next message free to expand it again
		isCollapsedByUser = isExpanded && !!(options.getValue() ?? '').trim();

		if (isExpanded && animatesSheet()) {
			void collapseSheet();

			return;
		}

		isExpanded = !isExpanded;
		// after the switch, so the editor redrawing as it takes focus does not reach into
		// the heights measured for it
		options.focusInput();
	}

	return {
		get isBlurHeld() {
			return isBlurHeld;
		},
		get isExpanded() {
			return isExpanded;
		},
		get isSheetMoving() {
			return isSheetMoving;
		},
		toggle
	};
}
