export enum ColorMode {
	DARK = 'dark',
	LIGHT = 'light',
	SYSTEM = 'system'
}

// From ui/apps/chat's src/lib/enums/ui.enums.ts: the enums the package's components use.
export enum TooltipSide {
	BOTTOM = 'bottom',
	LEFT = 'left',
	RIGHT = 'right',
	TOP = 'top'
}

// From ui/apps/chat's src/lib/enums/keyboard.enums.ts: the keys the components handle.
export enum KeyboardKey {
	ARROW_DOWN = 'ArrowDown',
	ARROW_LEFT = 'ArrowLeft',
	ARROW_RIGHT = 'ArrowRight',
	ARROW_UP = 'ArrowUp',
	END = 'End',
	HOME = 'Home'
}

/** Mouse buttons a pointer event reports (MouseEvent.button). */
export enum MouseButton {
	PRIMARY = 0
}

/** What a pointer event comes from (PointerEvent.pointerType). */
export enum PointerType {
	MOUSE = 'mouse',
	PEN = 'pen',
	TOUCH = 'touch'
}
