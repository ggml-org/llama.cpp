<script lang="ts">
	import {
		GENERIC_SKILL_PROVIDER_TOOLTIP,
		isGenericSkillProvider,
		skillProviderLabel
	} from './skill-presentation';
	import * as Tooltip from '$lib/components/ui/tooltip';

	interface Props {
		provider: string;
	}

	let { provider }: Props = $props();

	const label = $derived(skillProviderLabel(provider));
	const isGenericProvider = $derived(isGenericSkillProvider(provider));
</script>

{#if isGenericProvider}
	<Tooltip.Provider
		><Tooltip.Root
			><Tooltip.Trigger>
				{#snippet child({ props })}<span {...props}>{label}</span>{/snippet}
			</Tooltip.Trigger><Tooltip.Content><p>{GENERIC_SKILL_PROVIDER_TOOLTIP}</p></Tooltip.Content
			></Tooltip.Root
		></Tooltip.Provider
	>
{:else}
	<span>{label}</span>
{/if}
