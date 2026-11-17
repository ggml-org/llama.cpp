<script>
	import icon from 'llama-brand/logo/svg/icon/icon-light.svg?raw';

	// llama-brand draws the icon in its ink; currentColor lets it follow the text colour.
	const logoMark = icon.replace(/fill="#[0-9a-f]{3,8}"/gi, 'fill="currentColor"');

	let { class: className = '', style = '' } = $props();
</script>

<div class={className} {style}>
	{@html logoMark}
</div>

<style>
	div :global(svg) {
		width: var(--size, 1rem);
		height: var(--size, 1rem);
	}
</style>
