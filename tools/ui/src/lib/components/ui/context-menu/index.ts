import Content from './context-menu-content.svelte';
import Group from './context-menu-group.svelte';
import Item from './context-menu-item.svelte';
import Label from './context-menu-label.svelte';
import Separator from './context-menu-separator.svelte';
import Shortcut from './context-menu-shortcut.svelte';
import SubContent from './context-menu-sub-content.svelte';
import SubTrigger from './context-menu-sub-trigger.svelte';
import Trigger from './context-menu-trigger.svelte';
import { ContextMenu as ContextMenuPrimitive } from 'bits-ui';
const Sub = ContextMenuPrimitive.Sub;
const Root = ContextMenuPrimitive.Root;

export {
	Content,
	Root as ContextMenu,
	Content as ContextMenuContent,
	Group as ContextMenuGroup,
	Item as ContextMenuItem,
	Label as ContextMenuLabel,
	Separator as ContextMenuSeparator,
	Shortcut as ContextMenuShortcut,
	Sub as ContextMenuSub,
	SubContent as ContextMenuSubContent,
	SubTrigger as ContextMenuSubTrigger,
	Trigger as ContextMenuTrigger,
	Group,
	Item,
	Label,
	Root,
	Separator,
	Shortcut,
	Sub,
	SubContent,
	SubTrigger,
	Trigger
};
