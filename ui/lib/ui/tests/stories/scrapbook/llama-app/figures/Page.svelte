<!--
	llama.app's landing figures in one direction, in the page's order under their section titles, so the set can be
	judged as a whole: the chat and the API on the shared models, the settings Llama picks, the three lightweight facts,
	and nothing to bundle.
-->
<script lang="ts">
	import Bundle from './Bundle.svelte';
	import Light from './Light.svelte';
	import MentalModel from './MentalModel.svelte';
	import Settings from './Settings.svelte';
	import './figures.css';

	let { arch = false }: { arch?: boolean } = $props();
</script>

<div class="flex flex-col gap-14 p-6 @3xl:p-12">
	<section class="flex flex-col gap-5">
		<h2 class="text-heading">Like OpenAI, but on your Mac</h2>
		<MentalModel {arch} />
	</section>

	<section class="grid items-center gap-6 @3xl:grid-cols-2">
		<div class="flex flex-col gap-3">
			<h2 class="text-heading">Nothing to set up</h2>
			<p class="text-muted-foreground">
				Llama checks your Mac and sets up each model to run well on it. You only pick which model to talk to.
			</p>
		</div>
		<Settings {arch} />
	</section>

	<section class="flex flex-col gap-5">
		<h2 class="text-heading">Light enough to forget it's there</h2>
		<Light {arch} />
	</section>

	<section class="flex flex-col gap-5">
		<h2 class="text-heading">Nothing to bundle</h2>
		<Bundle {arch} />
	</section>
</div>
