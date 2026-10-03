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

/** Close control of a dialog, a panel or the navigation rail on a phone: one tap target,
 *  one cross, one look. Each site adds the offsets that keep the cross on the 16px inset
 *  the titles sit on. */
export const PANEL_CLOSE_MOBILE_CLASS =
	'max-md:grid max-md:size-10 max-md:place-items-center max-md:rounded-full max-md:text-muted-foreground max-md:opacity-100 max-md:hover:bg-accent max-md:hover:text-accent-foreground';

/** Selects the focused chat-form input (either renderer) to restore focus after model actions. */
export const CHAT_INPUT_FOCUS_SELECTOR =
	'[data-slot="input-area"] textarea, [data-slot="input-area"] [contenteditable="true"]';

/** Search input of an open dropdown-menu, focused after the menu mounts. */
export const DROPDOWN_MENU_CONTENT_SEARCH_SELECTOR = '[data-slot="dropdown-menu-content"] input';

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

/** One filter toggle of the models manager toolbar. */
export const FILTER_TOGGLE_ITEM_CLASS =
	'bg-muted! border-border/30! shadow-none! dark:border-border/20! data-[state=on]:bg-muted-foreground/15! data-[state=on]:text-foreground! dark:data-[state=on]:bg-muted-foreground/25!';

/** Column grid shared by every row of the models manager table. Below md the context,
 *  status and actions columns are dropped, so a phone gives the whole width to the
 *  model id and floats the row's last cell at the row's end. */
export const MODEL_ROW_GRID_CLASS =
	'grid grid-cols-[minmax(0,1fr)] items-center gap-3 md:grid-cols-[minmax(0,1fr)_11rem_3rem_4.5rem] md:gap-4';

/** Last cell of a row: its actions menu, or its fold control. A phone floats it at the
 *  row's end, inside the clearance the first cell leaves in front of it. */
export const MODEL_ROW_TRAILING_CELL_CLASS =
	'max-md:absolute max-md:top-1/2 max-md:right-2 max-md:-translate-y-1/2';

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
