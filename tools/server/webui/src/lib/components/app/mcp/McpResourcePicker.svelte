<script lang="ts">
	/**
	 * McpResourcePicker - Wrapper component for DialogMcpResources
	 *
	 * This component re-exports DialogMcpResources for backward compatibility.
	 * New code should use DialogMcpResources directly from dialogs.
	 */
	import { DialogMcpResources } from '$lib/components/app';
	import type { MCPResourceInfo } from '$lib/types';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
		onAttach?: (resource: MCPResourceInfo) => void;
		preSelectedUri?: string;
	}

	let { open = $bindable(false), onOpenChange, onAttach, preSelectedUri }: Props = $props();
</script>

<DialogMcpResources bind:open {onOpenChange} {onAttach} {preSelectedUri} />
