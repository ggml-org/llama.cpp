/**
 * SidebarDrag - Pointer-driven drag and drop in the sidebar: a whole section
 * by its heading, or a conversation row into another section.
 *
 * The browser's own drag and drop only moves a ghost picture, and the element
 * jumps from where it started once dropped. Here the real section follows the
 * pointer while the others slide around it, and a dragged chat is a lifted
 * copy of its row; either glides from where it was let go into its new place.
 */

import { uiStore } from '$lib/stores/ui.svelte';
import { tick } from 'svelte';

/** A section that takes dropped chats, registered by its drop zone. */
export interface SidebarDropZone {
	/** Whether the chat is in the section already; dropping it there would change nothing. */
	holds: (convId: string) => boolean;
	onDrop: (convId: string) => Promise<void> | void;
}

/** Pointer travel before a press becomes a drag, so a click stays a click. */
const DRAG_THRESHOLD_PX = 4;
/** How long a dropped element takes to glide into its place. */
const SETTLE_MS = 200;
/** Distance from the sidebar's top or bottom edge where dragging scrolls it. */
const AUTO_SCROLL_EDGE_PX = 48;
/** Fastest auto-scroll, in pixels per frame, right at the edge. */
const AUTO_SCROLL_MAX_SPEED = 14;
const SETTLE_EASING = 'cubic-bezier(0.23, 1, 0.32, 1)';
const SECTION_SLOT_ATTR = 'data-sidebar-slot';
const SECTION_ATTR = 'data-sidebar-section';
const CONVERSATION_ROW_ATTR = 'data-conversation-row';

interface Gesture {
	kind: 'section' | 'conversation';
	id: string;
	/** The section slot, or the conversation row. */
	element: HTMLElement;
	scroller: HTMLElement | null;
	startX: number;
	startY: number;
	startScroll: number;
	startOffsetTop: number;
	startRect: DOMRect;
	x: number;
	y: number;
	active: boolean;
	overlay: HTMLElement | null;
}

function findScroller(element: HTMLElement): HTMLElement | null {
	for (let node = element.parentElement; node; node = node.parentElement) {
		const { overflowY } = getComputedStyle(node);

		if (overflowY === 'auto' || overflowY === 'scroll') return node;
	}

	return null;
}

class SidebarDragStore {
	/** Chat being dragged, out of its section while a copy of its row follows the pointer. */
	convId = $state<string | null>(null);
	/** Chat whose row is hidden while its copy glides into it. */
	landingConvId = $state<string | null>(null);
	/** Section being dragged, or gliding into its slot after the drop. */
	sectionId = $state<string | null>(null);
	/** Vertical offset of the dragged section from its slot. */
	sectionOffset = $state(0);
	/** Sections in the order they take while one is dragged among them. */
	sectionOrder = $state<string[] | null>(null);
	/** Whether the dragged section is gliding into its slot. */
	settling = $state(false);
	/** Section the dragged chat would drop into. */
	targetSectionId = $state<string | null>(null);

	#autoScroll = () => {
		const gesture = this.#gesture;

		if (!gesture?.active) return;

		const scroller = gesture.scroller;

		if (scroller) {
			const rect = scroller.getBoundingClientRect();
			const fromTop = gesture.y - rect.top;
			const fromBottom = rect.bottom - gesture.y;

			let speed = 0;

			if (fromTop < AUTO_SCROLL_EDGE_PX) {
				speed = -AUTO_SCROLL_MAX_SPEED * (1 - Math.max(fromTop, 0) / AUTO_SCROLL_EDGE_PX);
			} else if (fromBottom < AUTO_SCROLL_EDGE_PX) {
				speed = AUTO_SCROLL_MAX_SPEED * (1 - Math.max(fromBottom, 0) / AUTO_SCROLL_EDGE_PX);
			}

			if (speed !== 0) {
				scroller.scrollTop += speed;
				this.#update();
			}
		}

		this.#frame = requestAnimationFrame(this.#autoScroll);
	};
	#frame = 0;
	#gesture: Gesture | null = null;

	#handleCancel = () => {
		void this.#finish(false);
	};

	#handleKeydown = (event: KeyboardEvent) => {
		if (event.key !== 'Escape' || !this.#gesture?.active) return;

		event.preventDefault();
		void this.#finish(false);
	};

	#handleMove = (event: PointerEvent) => {
		const gesture = this.#gesture;

		if (!gesture) return;

		gesture.x = event.clientX;
		gesture.y = event.clientY;

		if (!gesture.active) {
			if (
				Math.hypot(event.clientX - gesture.startX, event.clientY - gesture.startY) <
				DRAG_THRESHOLD_PX
			) {
				return;
			}

			this.#activate(gesture);
		}

		this.#update();
	};

	#handleUp = () => {
		void this.#finish(true);
	};

	#zones: Record<string, SidebarDropZone> = {};

	/** Starts watching a press on a conversation row; it becomes a drag once the pointer travels. */
	pressConversation(event: PointerEvent, convId: string): void {
		const row = (event.currentTarget as HTMLElement).closest<HTMLElement>(
			`[${CONVERSATION_ROW_ATTR}]`
		);

		if (row) this.#press(event, 'conversation', convId, row);
	}

	/** Starts watching a press on a section heading; it becomes a drag once the pointer travels. */
	pressSection(event: PointerEvent, sectionId: string): void {
		const slot = (event.currentTarget as HTMLElement).closest<HTMLElement>(
			`[${SECTION_SLOT_ATTR}]`
		);

		if (slot) this.#press(event, 'section', sectionId, slot);
	}

	registerZone(sectionId: string, zone: SidebarDropZone): () => void {
		this.#zones[sectionId] = zone;

		return () => {
			if (this.#zones[sectionId] === zone) delete this.#zones[sectionId];
		};
	}

	#activate(gesture: Gesture): void {
		gesture.active = true;
		window.getSelection()?.removeAllRanges();
		document.body.style.userSelect = 'none';
		document.body.style.cursor = 'grabbing';
		// rows and headings under the pointer stop reacting to it (see app.css)
		document.body.dataset.sidebarDragging = '';

		if (gesture.kind === 'section') {
			this.sectionId = gesture.id;
			this.sectionOffset = 0;
			this.sectionOrder = this.#slotIds(gesture);
		} else {
			// the copy first: the row leaves its section as soon as the chat is dragged
			gesture.overlay = this.#createOverlay(gesture.element, gesture.startRect);
			this.convId = gesture.id;
		}

		this.#frame = requestAnimationFrame(this.#autoScroll);
	}

	/** A lifted copy of the row, above everything, that follows the pointer. */
	#createOverlay(row: HTMLElement, rect: DOMRect): HTMLElement {
		const overlay = document.createElement('div');

		overlay.className = 'sidebar-drag-overlay';
		Object.assign(overlay.style, {
			height: `${rect.height}px`,
			left: `${rect.left}px`,
			top: `${rect.top}px`,
			width: `${rect.width}px`
		});
		overlay.append(row.cloneNode(true));
		document.body.append(overlay);

		return overlay;
	}

	async #finish(drop: boolean): Promise<void> {
		const gesture = this.#gesture;

		this.#gesture = null;
		cancelAnimationFrame(this.#frame);
		window.removeEventListener('pointermove', this.#handleMove);
		window.removeEventListener('pointerup', this.#handleUp);
		window.removeEventListener('pointercancel', this.#handleCancel);
		window.removeEventListener('keydown', this.#handleKeydown);

		if (!gesture?.active) return;

		document.body.style.userSelect = '';
		document.body.style.cursor = '';
		delete document.body.dataset.sidebarDragging;
		this.#swallowNextClick();

		if (gesture.kind === 'section') {
			await this.#settleSection(drop);
		} else {
			await this.#settleConversation(gesture, drop);
		}
	}

	#press(event: PointerEvent, kind: Gesture['kind'], id: string, element: HTMLElement): void {
		// a mouse or a pen: a finger keeps scrolling the sidebar
		if (event.button !== 0 || event.pointerType === 'touch' || this.#gesture) return;

		const scroller = findScroller(element);

		this.#gesture = {
			active: false,
			element,
			id,
			kind,
			overlay: null,
			scroller,
			startOffsetTop: element.offsetTop,
			startRect: element.getBoundingClientRect(),
			startScroll: scroller?.scrollTop ?? 0,
			startX: event.clientX,
			startY: event.clientY,
			x: event.clientX,
			y: event.clientY
		};

		window.addEventListener('pointermove', this.#handleMove);
		window.addEventListener('pointerup', this.#handleUp);
		window.addEventListener('pointercancel', this.#handleCancel);
		window.addEventListener('keydown', this.#handleKeydown);
	}

	#scrollDelta(gesture: Gesture): number {
		return (gesture.scroller?.scrollTop ?? 0) - gesture.startScroll;
	}

	async #settleConversation(gesture: Gesture, drop: boolean): Promise<void> {
		const overlay = gesture.overlay;
		const targetId = this.targetSectionId;
		const zone = drop && targetId ? this.#zones[targetId] : undefined;

		this.targetSectionId = null;

		if (zone) await zone.onDrop(gesture.id);

		// the row slides into its new section, or back into its own, hidden
		// until the copy has glided into it
		this.landingConvId = gesture.id;
		this.convId = null;
		await tick();

		const row = document.querySelector<HTMLElement>(
			`[${CONVERSATION_ROW_ATTR}="${CSS.escape(gesture.id)}"]`
		);
		// a folded section, or one past its preview, shows no row to land in
		const destination = row?.getBoundingClientRect();

		if (overlay) {
			overlay.style.transition = `transform ${SETTLE_MS}ms ${SETTLE_EASING}, opacity ${SETTLE_MS}ms ease-out`;

			if (destination) {
				overlay.style.transform = `translate(${destination.left - gesture.startRect.left}px, ${
					destination.top - gesture.startRect.top
				}px)`;
			} else {
				overlay.style.opacity = '0';
			}

			await new Promise((resolve) => setTimeout(resolve, SETTLE_MS));
			overlay.remove();
		}

		this.landingConvId = null;
	}

	async #settleSection(drop: boolean): Promise<void> {
		const order = this.sectionOrder;

		if (drop && order) {
			uiStore.setSidebarSectionOrder(order);
		} else {
			this.sectionOrder = null;
			await tick();
		}

		// the section glides from where it was let go into its slot
		this.settling = true;
		await tick();
		this.sectionOffset = 0;
		await new Promise((resolve) => setTimeout(resolve, SETTLE_MS));

		this.sectionId = null;
		this.sectionOrder = null;
		this.settling = false;
	}

	#slotIds(gesture: Gesture): string[] {
		const list = gesture.element.parentElement;

		return list
			? [...list.querySelectorAll<HTMLElement>(`:scope > [${SECTION_SLOT_ATTR}]`)].map(
					(slot) => slot.getAttribute(SECTION_SLOT_ATTR)!
				)
			: [];
	}

	/** The press that ended the drag must not also click what it was let go over. */
	#swallowNextClick(): void {
		const swallow = (event: MouseEvent) => {
			event.preventDefault();
			event.stopPropagation();
		};

		window.addEventListener('click', swallow, { capture: true, once: true });
		// the click, if any, comes right after the release
		setTimeout(() => window.removeEventListener('click', swallow, { capture: true }), 0);
	}

	#update(): void {
		const gesture = this.#gesture;

		if (!gesture?.active) return;

		if (gesture.kind === 'section') {
			void this.#updateSection(gesture);
		} else {
			this.#updateConversation(gesture);
		}
	}

	#updateConversation(gesture: Gesture): void {
		if (gesture.overlay) {
			gesture.overlay.style.transform = `translate(${gesture.x - gesture.startX}px, ${
				gesture.y - gesture.startY
			}px)`;
		}

		const section = document
			.elementFromPoint(gesture.x, gesture.y)
			?.closest<HTMLElement>(`[${SECTION_ATTR}]`);
		const sectionId = section?.getAttribute(SECTION_ATTR) ?? null;
		const zone = sectionId ? this.#zones[sectionId] : undefined;

		this.targetSectionId = zone && !zone.holds(gesture.id) ? sectionId : null;
	}

	async #updateSection(gesture: Gesture): Promise<void> {
		const travel = gesture.y - gesture.startY + this.#scrollDelta(gesture);
		const height = gesture.element.offsetHeight;
		const center = gesture.startOffsetTop + travel + height / 2;
		const list = gesture.element.parentElement;

		if (list && this.sectionOrder) {
			// the dragged section goes below every other section whose middle it passed
			const others = [...list.querySelectorAll<HTMLElement>(`:scope > [${SECTION_SLOT_ATTR}]`)]
				.filter((slot) => slot !== gesture.element)
				.sort((a, b) => a.offsetTop - b.offsetTop);
			const index = others.filter((slot) => slot.offsetTop + slot.offsetHeight / 2 < center).length;
			const next = others.map((slot) => slot.getAttribute(SECTION_SLOT_ATTR)!);

			next.splice(index, 0, gesture.id);

			if (next.join() !== this.sectionOrder.join()) {
				this.sectionOrder = next;
				await tick();
			}
		}

		// the slot may have moved under the section: keep the section under the pointer
		this.sectionOffset = gesture.startOffsetTop + travel - gesture.element.offsetTop;
	}
}

export const sidebarDragStore = new SidebarDragStore();
