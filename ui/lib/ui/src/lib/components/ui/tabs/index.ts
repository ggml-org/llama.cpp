import Root from './tabs.svelte';
import Content from './tabs-content.svelte';
import List, { type TabsListVariant, tabsListVariants } from './tabs-list.svelte';
import Trigger from './tabs-trigger.svelte';

export {
	Root,
	Content,
	List,
	Trigger,
	tabsListVariants,
	type TabsListVariant,
	//
	Root as Tabs,
	Content as TabsContent,
	List as TabsList,
	Trigger as TabsTrigger
};
