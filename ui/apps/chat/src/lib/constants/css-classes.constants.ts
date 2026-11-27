export {
	BOX_BORDER,
	FILTER_TOGGLE_ITEM_CLASS,
	FILTER_TRIGGER_CLASS,
	ICON_CLASS_DEFAULT,
	INPUT_CLASSES,
	PANEL_CLOSE_MOBILE_CLASS
} from '@ggml-org/llama-ui/constants';
export { MODEL_BADGE_CLASS, MODEL_VARIANT_BADGE_CLASS } from '@ggml-org/llama-ui/models';

export const PANEL_CLASSES = `
    bg-background
    elevated-control backdrop-blur-lg!
    rounded-t-lg!
`;

export const CHAT_FORM_POPOVER_MAX_HEIGHT = 'max-h-80';
export const DIALOG_SUBMENU_CONTENT = 'w-60';

/** Shape of a chat tab, shared with the breadcrumb items next to the project tabs. */
export const TAB_SURFACE_CLASS =
	'flex h-8 items-center rounded-lg border text-body-sm whitespace-nowrap backdrop-blur-xl';

/** Tab surface at rest: invisible until hovered. */
export const TAB_SURFACE_IDLE_CLASS =
	'border-transparent hover:bg-primary/10 hover:border-border/10 hover:shadow-(--elevation-control)';

/** A round matte glass key (see GlassKey): white glass over the page, a faint hairline, and
 *  matte-key's blur and saturation with its small shadow, so what scrolls under it softens. */
export const GLASS_KEY_CLASS =
	'matte-key flex shrink-0 cursor-pointer items-center justify-center rounded-full border border-border/20 bg-(--surface-glass) text-accent-foreground outline-none transition-colors focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50';

/** Tab surface of the open tab. */
export const TAB_SURFACE_ACTIVE_CLASS =
	'matte-key bg-muted/60 border-border/10 text-accent-foreground hover:bg-primary/15';

/** Selects the focused chat-form input (either renderer) to restore focus after model actions. */
export const CHAT_INPUT_FOCUS_SELECTOR =
	'[data-slot="input-area"] textarea, [data-slot="input-area"] [contenteditable="true"]';

/** Search input of an open dropdown-menu, focused after the menu mounts. */
export const DROPDOWN_MENU_CONTENT_SEARCH_SELECTOR = '[data-slot="dropdown-menu-content"] input';

/**
 * Sticky offset of a model-list group heading under the dropdown's search block:
 * the search header exposes its height as `--dropdown-sticky-height`, and the
 * heading overlaps the header's bottom border, so the two stack without a gap.
 */
export const SELECTOR_GROUP_STICKY_OFFSET = 'top: calc(var(--dropdown-sticky-height, 0px) - 1px)';

/** Column grid shared by every row of the models manager table. Below md the context,
 *  status and actions columns are dropped, so a phone gives the whole width to the
 *  model id and floats the row's last cell at the row's end. */
export const MODEL_ROW_GRID_CLASS =
	'grid grid-cols-[minmax(0,1fr)] items-center gap-3 md:grid-cols-[minmax(0,1fr)_11rem_3rem_4.5rem] md:gap-4';

/** Last cell of a row: its actions menu, or its fold control. A phone floats it at the
 *  row's end, inside the clearance the first cell leaves in front of it. */
export const MODEL_ROW_TRAILING_CELL_CLASS =
	'max-md:absolute max-md:top-1/2 max-md:right-3 max-md:-translate-y-1/2';

/** Small Tailwind size class for inline icons. */
export const ICON_CLASS_SM = 'size-icon-sm';

/** Extra-small Tailwind size class for inline icons. */
export const ICON_CLASS_XS = 'size-icon-xs';
