// only the classes the copied components import, from ui/apps/chat's css-classes.constants.ts

export const BOX_BORDER =
	'border border-border/30 focus-within:border-border  dark:border-border/20 dark:focus-within:border-border';

export const INPUT_CLASSES = `
    bg-muted/60 dark:bg-muted/75
    ${BOX_BORDER}
    matte-key
    outline-none
    text-foreground
`;

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

/** One filter toggle of the models manager toolbar: a 32px wide key in the 32px tall group. */
export const FILTER_TOGGLE_ITEM_CLASS = 'w-8';

export const ICON_CLASS_DEFAULT = 'size-icon-md';

/** Close control of a dialog, a panel or the navigation rail on a phone: one tap target,
 *  one cross, one look. Each site adds the offsets that keep the cross on the 16px inset
 *  the titles sit on. */
export const PANEL_CLOSE_MOBILE_CLASS =
	'max-md:grid max-md:size-10 max-md:place-items-center max-md:rounded-full max-md:text-muted-foreground max-md:opacity-100 max-md:hover:bg-accent max-md:hover:text-accent-foreground';
