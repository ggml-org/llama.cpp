import { describe, expect, it } from 'vitest';
import { getJpegOrientationFromDataURL, isJpegMimeType } from '$lib/utils/jpeg-orientation';

// Builds the TIFF payload of an APP1 segment holding a single IFD0 entry
function buildTiff(littleEndian: boolean, tag: number, value: number): number[] {
	const u16 = (v: number) => (littleEndian ? [v & 0xff, v >> 8] : [v >> 8, v & 0xff]);
	const u32 = (v: number) =>
		littleEndian
			? [v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff]
			: [(v >> 24) & 0xff, (v >> 16) & 0xff, (v >> 8) & 0xff, v & 0xff];

	return [
		...(littleEndian ? [0x49, 0x49] : [0x4d, 0x4d]),
		...u16(42),
		...u32(8),
		...u16(1),
		...u16(tag),
		...u16(3),
		...u32(1),
		// SHORT value sits left justified in the 4 byte value field
		...u16(value),
		...u16(0),
		...u32(0)
	];
}

// Wraps a TIFF payload into a complete minimal JPEG data URL
function buildJpegDataURL(tiff: number[] | null, prependApp0 = false): string {
	const bytes: number[] = [0xff, 0xd8];

	if (prependApp0) {
		// JFIF APP0 segment, irrelevant content the parser walks over
		bytes.push(0xff, 0xe0, 0x00, 0x07, 0x4a, 0x46, 0x49, 0x46, 0x00);
	}

	if (tiff) {
		const payload = [0x45, 0x78, 0x69, 0x66, 0x00, 0x00, ...tiff];
		const length = payload.length + 2;
		bytes.push(0xff, 0xe1, length >> 8, length & 0xff, ...payload);
	}

	// SOS marker terminates the metadata scan
	bytes.push(0xff, 0xda, 0x00, 0x02);

	return `data:image/jpeg;base64,${btoa(String.fromCharCode(...bytes))}`;
}

describe('getJpegOrientationFromDataURL', () => {
	it('returns the orientation from a little endian EXIF block', () => {
		expect(getJpegOrientationFromDataURL(buildJpegDataURL(buildTiff(true, 0x0112, 6)))).toBe(6);
	});

	it('returns the orientation from a big endian EXIF block', () => {
		expect(getJpegOrientationFromDataURL(buildJpegDataURL(buildTiff(false, 0x0112, 8)))).toBe(8);
	});

	it('walks over a leading APP0 segment', () => {
		expect(getJpegOrientationFromDataURL(buildJpegDataURL(buildTiff(true, 0x0112, 3), true))).toBe(
			3
		);
	});

	it('returns 1 when the EXIF block holds no orientation tag', () => {
		expect(getJpegOrientationFromDataURL(buildJpegDataURL(buildTiff(true, 0x0100, 6)))).toBe(1);
	});

	it('returns 1 when the orientation value is out of range', () => {
		expect(getJpegOrientationFromDataURL(buildJpegDataURL(buildTiff(true, 0x0112, 9)))).toBe(1);
	});

	it('returns 1 when the JPEG has no APP1 segment', () => {
		expect(getJpegOrientationFromDataURL(buildJpegDataURL(null, true))).toBe(1);
	});

	it('returns 1 for a payload that is not a JPEG', () => {
		const png = btoa(String.fromCharCode(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a));
		expect(getJpegOrientationFromDataURL(`data:image/png;base64,${png}`)).toBe(1);
	});

	it('returns 1 for a truncated payload', () => {
		const truncated = btoa(String.fromCharCode(0xff, 0xd8, 0xff));
		expect(getJpegOrientationFromDataURL(`data:image/jpeg;base64,${truncated}`)).toBe(1);
	});

	it('returns 1 for a malformed data URL', () => {
		expect(getJpegOrientationFromDataURL('not a data url')).toBe(1);
	});
});

describe('isJpegMimeType', () => {
	it('matches both JPEG MIME variants', () => {
		expect(isJpegMimeType('image/jpeg')).toBe(true);
		expect(isJpegMimeType('image/jpg')).toBe(true);
	});

	it('rejects other image MIME types', () => {
		expect(isJpegMimeType('image/png')).toBe(false);
		expect(isJpegMimeType('image/webp')).toBe(false);
	});
});
