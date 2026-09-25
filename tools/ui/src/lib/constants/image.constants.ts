/** Image handling constants */

export const IMAGE = {
	/** JPEG quality used when transcoding HEIC images. */
	HEIC_JPEG_QUALITY: 0.85,
	/** Unit conversion: pixels per megapixel. */
	MEGAPIXELS_TO_PIXELS: 1_000_000
} as const;

/**
 * JPEG and EXIF binary format constants for orientation parsing.
 */
export const EXIF = {
	/** APP1 segment marker byte, carries the EXIF payload */
	APP1_MARKER: 0xe1,
	/** "Exif" signature opening the APP1 payload, big endian uint32 */
	EXIF_SIGNATURE: 0x45786966,
	/** Size in bytes of one IFD directory entry */
	IFD_ENTRY_SIZE: 12,
	/** JPEG start of image marker */
	JPEG_SOI_MARKER: 0xffd8,
	/** EXIF tag id holding the orientation value */
	ORIENTATION_TAG: 0x0112,
	/** Bytes of file prefix to scan, the APP1 EXIF segment sits near the start */
	SCAN_BYTE_LIMIT: 128 * 1024,
	/** Start of scan marker byte, compressed data begins and no EXIF follows */
	SOS_MARKER: 0xda,
	/** TIFF byte order mark for little endian ("II") */
	TIFF_LITTLE_ENDIAN: 0x4949,
	/** TIFF magic number following the byte order mark */
	TIFF_MAGIC: 42
} as const;
