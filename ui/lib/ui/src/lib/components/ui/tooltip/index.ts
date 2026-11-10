import Content from './tooltip-content.svelte';
import Trigger from './tooltip-trigger.svelte';
import { Tooltip as TooltipPrimitive } from 'bits-ui';

// annotated so svelte-package can name the types in the generated declarations (llama-ui only)
const Root: typeof TooltipPrimitive.Root = TooltipPrimitive.Root;
const Provider: typeof TooltipPrimitive.Provider = TooltipPrimitive.Provider;
const Portal: typeof TooltipPrimitive.Portal = TooltipPrimitive.Portal;

export {
	Root,
	Trigger,
	Content,
	Provider,
	Portal,
	//
	Root as Tooltip,
	Content as TooltipContent,
	Trigger as TooltipTrigger,
	Provider as TooltipProvider,
	Portal as TooltipPortal
};
