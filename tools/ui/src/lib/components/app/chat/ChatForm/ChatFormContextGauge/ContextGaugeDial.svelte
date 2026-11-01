<script lang="ts">
	import { colorLevelTextClass } from './context-gauge';
	import { ColorLevel } from '$lib/enums';

	interface Props {
		percent: number | null;
		level: ColorLevel;
		/** sm inline, md a phone's inline touch target, key the icon of a phone's corner key */
		size?: 'sm' | 'md' | 'key';
	}

	let { level, percent, size = 'sm' }: Props = $props();

	// The dials draw on a 32 grid. The key dial draws on the icon grid instead (24, a 2px
	// stroke), as wide as the menu key's glyph across from it, so the two keys read alike;
	// the md dial fills its box, as the ring would sit inside a padding ring otherwise
	const GRID = $derived(size === 'key' ? 24 : 32);
	const RADIUS = $derived(size === 'key' ? 8.5 : size === 'md' ? 13 : 11);
	const CIRCUMFERENCE = $derived(2 * Math.PI * RADIUS);

	// on the key, at rest, the arc takes the icon colour of the key, not the muted one
	const strokeLevelClass = $derived(
		size === 'key' && level !== ColorLevel.WARNING && level !== ColorLevel.CRITICAL
			? 'text-accent-foreground'
			: colorLevelTextClass(level)
	);
	const dimensions = $derived(size === 'sm' ? 'size-5' : 'size-6');
	const strokeWidth = $derived(size === 'key' ? 2 : size === 'md' ? 4 : 3);
	const trackOpacity = $derived(size === 'key' ? 0.2 : 0.1);
</script>

<svg class={dimensions} fill="none" viewBox="0 0 {GRID} {GRID}">
	<circle
		cx={GRID / 2}
		cy={GRID / 2}
		r={RADIUS}
		stroke="currentColor"
		stroke-opacity={trackOpacity}
		stroke-width={strokeWidth}
	/>

	<circle
		class="transition-colors duration-300 {strokeLevelClass}"
		cx={GRID / 2}
		cy={GRID / 2}
		r={RADIUS}
		stroke="currentColor"
		stroke-dasharray={CIRCUMFERENCE}
		stroke-dashoffset={percent !== null ? CIRCUMFERENCE * (1 - percent / 100) : CIRCUMFERENCE}
		stroke-linecap="round"
		stroke-width={strokeWidth}
		transform="rotate(-90 {GRID / 2} {GRID / 2})"
	/>
</svg>
