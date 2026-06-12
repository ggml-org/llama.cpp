import {
	EXIF_SCAN_BYTE_LIMIT,
	JPEG_SOI_MARKER,
	APP1_MARKER,
	SOS_MARKER,
	EXIF_SIGNATURE,
	TIFF_LITTLE_ENDIAN,
	TIFF_MAGIC,
	EXIF_ORIENTATION_TAG,
	IFD_ENTRY_SIZE
} from '$lib/constants/jpeg-exif';
import { MimeTypeImage } from '$lib/enums';

/**
 * Read the EXIF orientation tag from a JPEG base64 data URL
 *
 * Only a bounded prefix of the base64 payload is decoded, the APP1 segment
 * always sits near the start of the file.
 * @param base64UrlJpeg - The JPEG base64 data URL to inspect
 * @returns The orientation value (1 to 8), or 1 when absent or unreadable
 */
export function getJpegOrientationFromDataURL(base64UrlJpeg: string): number {
	try {
		const payloadStart = base64UrlJpeg.indexOf(',') + 1;

		if (payloadStart <= 0) {
			return 1;
		}

		// Keep the slice a multiple of 4 characters so atob accepts it
		const charLimit = Math.ceil(EXIF_SCAN_BYTE_LIMIT / 3) * 4;
		const slice = base64UrlJpeg.slice(payloadStart, payloadStart + charLimit);
		const binary = atob(slice.slice(0, slice.length - (slice.length % 4)));
		const bytes = new Uint8Array(binary.length);

		for (let i = 0; i < binary.length; i++) {
			bytes[i] = binary.charCodeAt(i);
		}

		return findExifOrientation(new DataView(bytes.buffer));
	} catch {
		return 1;
	}
}

/**
 * Walk the JPEG segments of a header buffer looking for the APP1 EXIF block
 * @param view - DataView over the JPEG header bytes
 * @returns The orientation value (1 to 8), or 1 when absent or malformed
 */
function findExifOrientation(view: DataView): number {
	if (view.byteLength < 4 || view.getUint16(0) !== JPEG_SOI_MARKER) {
		return 1;
	}

	let offset = 2;

	while (offset + 4 <= view.byteLength) {
		if (view.getUint8(offset) !== 0xff) {
			return 1;
		}

		const marker = view.getUint8(offset + 1);

		// Compressed image data starts here: no EXIF past this point
		if (marker === SOS_MARKER) {
			return 1;
		}

		const segmentLength = view.getUint16(offset + 2);

		if (marker === APP1_MARKER) {
			return parseExifOrientation(view, offset + 4, segmentLength);
		}

		offset += 2 + segmentLength;
	}

	return 1;
}

/**
 * Parse the orientation tag from an APP1 EXIF payload
 * @param view - DataView over the JPEG header bytes
 * @param start - Offset of the APP1 payload, right after the segment length
 * @param segmentLength - Declared APP1 segment length
 * @returns The orientation value (1 to 8), or 1 when absent or malformed
 */
function parseExifOrientation(view: DataView, start: number, segmentLength: number): number {
	const end = Math.min(start + segmentLength, view.byteLength);

	// The payload opens with the "Exif\0\0" signature
	if (
		start + 6 > end ||
		view.getUint32(start) !== EXIF_SIGNATURE ||
		view.getUint16(start + 4) !== 0
	) {
		return 1;
	}

	const tiff = start + 6;

	if (tiff + 8 > end) {
		return 1;
	}

	const littleEndian = view.getUint16(tiff) === TIFF_LITTLE_ENDIAN;

	if (view.getUint16(tiff + 2, littleEndian) !== TIFF_MAGIC) {
		return 1;
	}

	const ifdOffset = view.getUint32(tiff + 4, littleEndian);

	if (tiff + ifdOffset + 2 > end) {
		return 1;
	}

	const entryCount = view.getUint16(tiff + ifdOffset, littleEndian);

	// Scan IFD0 entries for the orientation tag
	for (let i = 0; i < entryCount; i++) {
		const entry = tiff + ifdOffset + 2 + i * IFD_ENTRY_SIZE;

		if (entry + IFD_ENTRY_SIZE > end) {
			return 1;
		}

		if (view.getUint16(entry, littleEndian) === EXIF_ORIENTATION_TAG) {
			const orientation = view.getUint16(entry + 8, littleEndian);

			return orientation >= 1 && orientation <= 8 ? orientation : 1;
		}
	}

	return 1;
}

/**
 * Check if a MIME type represents a JPEG
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type is a JPEG variant
 */
export function isJpegMimeType(mimeType: string): boolean {
	return mimeType === MimeTypeImage.JPEG || mimeType === MimeTypeImage.JPG;
}
