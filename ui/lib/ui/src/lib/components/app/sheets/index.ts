/** A rounded group of rows with an optional title and caption, as a phone's settings list them. */
export { default as InsetList } from './InsetList.svelte';

/** A row of an InsetList: a button, a label around a switch, or a row holding a control. */
export { default as InsetListRow } from './InsetListRow.svelte';

/** A row of an InsetList that opens a page and has a control of its own at its end. */
export { default as InsetListSplitRow } from './InsetListSplitRow.svelte';

/** A sheet's header: the close or back key in its corner and the centered title. */
export { default as SheetHeader } from './SheetHeader.svelte';

/** The scrolling body of a sheet, sliding between the pages of a `useSheetPages` stack. */
export { default as SheetPages } from './SheetPages.svelte';

export { type SheetPagesState, useSheetPages } from './sheet-pages.svelte';
