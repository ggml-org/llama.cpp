/**
 * Reads the metadata key/value pairs at the start of a GGUF file, from as many bytes of it as are given.
 * Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */

const GGUF_MAGIC = 0x46554747; // "GGUF", little-endian

enum GgufType {
	UINT8 = 0,
	INT8 = 1,
	UINT16 = 2,
	INT16 = 3,
	UINT32 = 4,
	INT32 = 5,
	FLOAT32 = 6,
	BOOL = 7,
	STRING = 8,
	ARRAY = 9,
	UINT64 = 10,
	INT64 = 11,
	FLOAT64 = 12
}

const SCALAR_SIZE: Partial<Record<GgufType, number>> = {
	[GgufType.BOOL]: 1,
	[GgufType.FLOAT32]: 4,
	[GgufType.FLOAT64]: 8,
	[GgufType.INT8]: 1,
	[GgufType.INT16]: 2,
	[GgufType.INT32]: 4,
	[GgufType.INT64]: 8,
	[GgufType.UINT8]: 1,
	[GgufType.UINT16]: 2,
	[GgufType.UINT32]: 4,
	[GgufType.UINT64]: 8
};

export type GgufValue = boolean | number | bigint | string;

export interface GgufMetadata {
	/** True when every pair was read; false when the buffer ended first or is not GGUF. */
	complete: boolean;
	metadata: Map<string, GgufValue>;
}

/**
 * The scalar and string metadata of a GGUF file (arrays are skipped). A buffer cut short returns the pairs read
 * before the cut; a buffer that is not GGUF returns an empty map.
 */
export function readGgufMetadata(buffer: ArrayBuffer): GgufMetadata {
	const view = new DataView(buffer);
	const decoder = new TextDecoder();
	const metadata = new Map<string, GgufValue>();

	let offset = 0;

	const fits = (bytes: number) => offset + bytes <= view.byteLength;
	const readString = (): string | null => {
		if (!fits(8)) return null;

		const length = Number(view.getBigUint64(offset, true));

		if (!fits(8 + length)) return null;

		const text = decoder.decode(new Uint8Array(buffer, offset + 8, length));

		offset += 8 + length;

		return text;
	};
	const readScalar = (type: GgufType): GgufValue | null => {
		const size = SCALAR_SIZE[type];

		if (size === undefined || !fits(size)) return null;

		const at = offset;

		offset += size;

		switch (type) {
			case GgufType.UINT8:
				return view.getUint8(at);
			case GgufType.INT8:
				return view.getInt8(at);
			case GgufType.UINT16:
				return view.getUint16(at, true);
			case GgufType.INT16:
				return view.getInt16(at, true);
			case GgufType.UINT32:
				return view.getUint32(at, true);
			case GgufType.INT32:
				return view.getInt32(at, true);
			case GgufType.FLOAT32:
				return view.getFloat32(at, true);
			case GgufType.BOOL:
				return view.getUint8(at) !== 0;
			case GgufType.UINT64:
				return view.getBigUint64(at, true);
			case GgufType.INT64:
				return view.getBigInt64(at, true);
			default:
				return view.getFloat64(at, true);
		}
	};
	// moves past an array's elements; false when the buffer ends inside it
	const skipArray = (): boolean => {
		if (!fits(12)) return false;

		const type = view.getUint32(offset, true) as GgufType;
		const count = Number(view.getBigUint64(offset + 4, true));

		offset += 12;

		for (let i = 0; i < count; i++) {
			const value = type === GgufType.STRING ? readString() : readScalar(type);

			if (value === null) return false;
		}

		return true;
	};

	// magic, version, tensor count, metadata count
	if (!fits(24) || view.getUint32(0, true) !== GGUF_MAGIC) return { complete: false, metadata };

	const count = Number(view.getBigUint64(16, true));

	offset = 24;

	for (let i = 0; i < count; i++) {
		const key = readString();

		if (key === null || !fits(4)) return { complete: false, metadata };

		const type = view.getUint32(offset, true) as GgufType;

		offset += 4;

		if (type === GgufType.ARRAY) {
			if (!skipArray()) return { complete: false, metadata };

			continue;
		}

		const value = type === GgufType.STRING ? readString() : readScalar(type);

		if (value === null) return { complete: false, metadata };

		metadata.set(key, value);
	}

	return { complete: true, metadata };
}
