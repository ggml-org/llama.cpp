<!-- A grid of <Figure>s: "steps" fit the origin's six tiles on one row where they can, "rules" lay the six rule
     panels out 3 x 2 on wide screens, 2 x 3 on tablets and in one column on phones. -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	let { children, layout = 'rules' }: { layout?: 'steps' | 'rules'; children: Snippet } = $props();
</script>

<div class={['figures', layout]}>{@render children()}</div>

<style>
	.figures {
		display: grid;
		gap: 16px;
	}
	.steps {
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
	}
	.rules {
		grid-template-columns: repeat(3, minmax(0, 1fr));
	}
	@media (max-width: 860px) {
		.rules {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	@media (max-width: 520px) {
		.rules {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
