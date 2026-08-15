<script lang="ts">
	import SkillsPage from '../../../src/routes/skills/+page.svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
</script>

<!--
	Test wrapper that provides the Tooltip provider the root layout installs,
	so desktop-width renders (where ActionIcon tooltips are active) work.
-->
<Tooltip.Provider>
	<SkillsPage />
</Tooltip.Provider>
