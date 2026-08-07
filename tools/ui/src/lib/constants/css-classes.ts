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

/** Default Tailwind size class for inline icon components (lucide, etc.). */
export const ICON_CLASS_DEFAULT = 'h-4 w-4';

/** Icon size + spinning animation; used for live-streaming tool indicators. */
export const ICON_CLASS_SPIN = 'h-4 w-4 animate-spin';
