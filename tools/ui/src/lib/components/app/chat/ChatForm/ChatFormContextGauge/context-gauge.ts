import { ColorLevel } from '$lib/enums';

const WARNING_THRESHOLD = 80;
const CRITICAL_THRESHOLD = 95;

export function colorLevelFromPercent(percent: number | null): ColorLevel {
	if (percent === null) return ColorLevel.NEUTRAL;

	if (percent >= CRITICAL_THRESHOLD) return ColorLevel.CRITICAL;

	if (percent >= WARNING_THRESHOLD) return ColorLevel.WARNING;

	return ColorLevel.OK;
}

export function colorLevelTextClass(level: ColorLevel): string {
	switch (level) {
		case ColorLevel.CRITICAL:
			return 'text-fg-danger';
		case ColorLevel.WARNING:
			return 'text-fg-warning';
		case ColorLevel.OK:
			return 'text-muted-foreground';
		default:
			return 'text-muted-foreground';
	}
}

export function colorLevelBgClass(level: ColorLevel): string {
	switch (level) {
		case ColorLevel.CRITICAL:
			return 'bg-fill-danger';
		case ColorLevel.WARNING:
			return 'bg-fill-warning';
		case ColorLevel.OK:
			return 'bg-fill-success';
		default:
			return 'bg-muted';
	}
}
