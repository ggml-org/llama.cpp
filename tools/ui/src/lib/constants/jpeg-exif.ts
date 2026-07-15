/**
 * JPEG and EXIF binary format constants for orientation parsing.
 */

/** Bytes of file prefix to scan, the APP1 EXIF segment sits near the start */
export const EXIF_SCAN_BYTE_LIMIT = 128 * 1024;

/** JPEG start of image marker */
export const JPEG_SOI_MARKER = 0xffd8;

/** APP1 segment marker byte, carries the EXIF payload */
export const APP1_MARKER = 0xe1;

/** Start of scan marker byte, compressed data begins and no EXIF follows */
export const SOS_MARKER = 0xda;

/** "Exif" signature opening the APP1 payload, big endian uint32 */
export const EXIF_SIGNATURE = 0x45786966;

/** TIFF byte order mark for little endian ("II") */
export const TIFF_LITTLE_ENDIAN = 0x4949;

/** TIFF magic number following the byte order mark */
export const TIFF_MAGIC = 42;

/** EXIF tag id holding the orientation value */
export const EXIF_ORIENTATION_TAG = 0x0112;

/** Size in bytes of one IFD directory entry */
export const IFD_ENTRY_SIZE = 12;
