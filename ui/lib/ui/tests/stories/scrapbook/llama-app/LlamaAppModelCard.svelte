<!--
	A catalog family as ModelFamilyCard: what llama.app passes it from its catalog and what it reads from the Hub.
-->
<script lang="ts">
	import type { Family } from './catalog';
	import {
		familyMem,
		modalitiesOf,
		reasons,
		releasedFor,
		sizeRange,
		sizesSummary,
		usesTools
	} from './catalog-helpers';
	import Description from './Description.svelte';
	import { ModelFamilyCard } from '$lib/components/models';

	let { class: className = '', family, href }: { class?: string; family: Family; href?: string } = $props();
</script>

<ModelFamilyCard
	class={className}
	{href}
	memGb={familyMem(family)}
	modalities={modalitiesOf(family.sizes)}
	name={family.name}
	nonCommercial={family.nonCommercial}
	org={family.org}
	quantOrg="ggml-org"
	released={releasedFor(family)}
	sizeRange={sizeRange(family.sizes)}
	sizes={sizesSummary(family)}
	supportsThinking={reasons(family.sizes)}
	supportsToolUse={usesTools(family.sizes)}
>
	<Description text={family.description} />
</ModelFamilyCard>
