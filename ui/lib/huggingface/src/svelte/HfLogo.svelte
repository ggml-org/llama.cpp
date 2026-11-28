<script lang="ts">
	import hfLogo from './assets/hf-logo.svg?url';

	interface Props {
		class?: string;
	}

	let { class: className = 'size-4' }: Props = $props();
</script>

<img alt="" class={className} src={hfLogo} />
