export const MS_PER_SECOND = 1000;
export const SECONDS_PER_MINUTE = 60;
export const SECONDS_PER_HOUR = 3600;
export const SHORT_DURATION_THRESHOLD = 1;
export const MEDIUM_DURATION_THRESHOLD = 10;

/** Default display value when no performance time is available */
export const DEFAULT_PERFORMANCE_TIME = '0s';

/** Max length before reasoning preview is truncated */
export const MAX_PREVIEW_LENGTH = 120;

export const STRIP_MARKDOWN_CAPTURE_PATTERNS: [RegExp, string][] = [
	[/^```(.*)/gm, '$1'],
	[/(.*)```$/gm, '$1'],
	[/`([^`]*)`/g, '$1'],
	[/\*\*(.*?)\*\*/g, '$1'],
	[/__(.*?)__/g, '$1'],
	[/\*(.*?)\*/g, '$1'],
	[/_(.*?)_/g, '$1']
];

/* eslint-disable no-misleading-character-class */
export const STRIP_MARKDOWN_INLINE_REGEX = new RegExp(
	[
		'<[^>]*>',
		'^>\\s*',
		'^#{1,6}\\s+',
		'^[\\s]*[-*+]\\s+',
		'^[\\s]*\\d+[.)]\\s+',
		'[\\u{1F600}-\\u{1F64F}\\u{1F300}-\\u{1F5FF}\\u{1F680}-\\u{1F6FF}\\u{1F1E0}-\\u{1F1FF}\\u{2600}-\\u{26FF}\\u{2700}-\\u{27BF}\\u{FE00}-\\u{FE0F}\\u{1F900}-\\u{1F9FF}\\u{1FA00}-\\u{1FA6F}\\u{1FA70}-\\u{1FAFF}\\u{200D}\\u{20E3}\\u{231A}-\\u{231B}\\u{23E9}-\\u{23F3}\\u{23F8}-\\u{23FA}\\u{25AA}-\\u{25AB}\\u{25B6}\\u{25C0}\\u{25FB}-\\u{25FE}\\u{2934}-\\u{2935}\\u{2B05}-\\u{2B07}\\u{2B1B}-\\u{2B1C}\\u{2B50}\\u{2B55}\\u{3030}\\u{303D}\\u{3297}\\u{3299}]'
	].join('|'),
	'gmu'
);
/* eslint-enable no-misleading-character-class */
