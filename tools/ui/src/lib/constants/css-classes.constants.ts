export const BOX_BORDER =
	'border border-border/30 focus-within:border-border  dark:border-border/20 dark:focus-within:border-border';

export const INPUT_CLASSES = `
    bg-muted/60 dark:bg-muted/75
    ${BOX_BORDER}
    shadow-sm
    outline-none
    text-foreground
`;

export const PANEL_CLASSES = `
    bg-background
    border border-border/30 dark:border-border/20
    shadow-sm backdrop-blur-lg!
    rounded-t-lg!
`;

export const CHAT_FORM_POPOVER_MAX_HEIGHT = 'max-h-80';
export const DIALOG_SUBMENU_CONTENT = 'w-60';

/** Selects the focused chat-form input (either renderer) to restore focus after model actions. */
export const CHAT_INPUT_FOCUS_SELECTOR =
	'[data-slot="input-area"] textarea, [data-slot="input-area"] [contenteditable="true"]';

/** Filter controls above the model table: one box, one height, one fill. */
export const FILTER_TRIGGER_CLASS = `
    h-8
    gap-1.5
    rounded-md
    px-3
    text-sm
    font-medium
    transition-colors
    hover:bg-muted/80 dark:hover:bg-muted
    ${INPUT_CLASSES}
`;

/** Column grid shared by every row of the models manager table. */
export const MODEL_ROW_GRID_CLASS =
	'grid grid-cols-[minmax(0,1fr)_11rem_3rem_4.5rem] items-center gap-4';

/** Neutral model badge: params, quantization, tags. */
export const MODEL_BADGE_CLASS =
	'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono bg-foreground/15 dark:bg-foreground/10 text-foreground [a&]:hover:bg-foreground/25';

/** Emphasis model badge: draft sidecars and other markers that must stand out. */
export const MODEL_VARIANT_BADGE_CLASS =
	'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md bg-primary px-1.5 py-0 text-[10px] font-mono font-semibold uppercase tracking-wide text-primary-foreground';

/** Default Tailwind size class for inline icon components (lucide, etc.). */
export const ICON_CLASS_DEFAULT = 'h-4 w-4';

/** Small Tailwind size class for inline icons. */
export const ICON_CLASS_SM = 'h-3.5 w-3.5';

/** Extra-small Tailwind size class for inline icons. */
export const ICON_CLASS_XS = 'h-3 w-3';

/** Icon size + spinning animation; used for live-streaming tool indicators. */
export const ICON_CLASS_SPIN = 'h-4 w-4 animate-spin';
