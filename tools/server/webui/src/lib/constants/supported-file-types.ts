/**
 * Comprehensive dictionary of all supported file types in webui
 * Organized by category with TypeScript enums for better type safety
 */

// File type category enum
export enum FileTypeCategory {
	IMAGE = 'image',
	AUDIO = 'audio',
	PDF = 'pdf',
	TEXT = 'text'
}

// Specific file type enums for each category
export enum ImageFileType {
	JPEG = 'jpeg',
	PNG = 'png',
	GIF = 'gif',
	WEBP = 'webp',
	SVG = 'svg'
}

export enum AudioFileType {
	MP3 = 'mp3',
	WAV = 'wav',
	WEBM = 'webm'
}

export enum PdfFileType {
	PDF = 'pdf'
}

export enum TextFileType {
	PLAIN_TEXT = 'plainText',
	MARKDOWN = 'markdown',
	JAVASCRIPT = 'javascript',
	TYPESCRIPT = 'typescript',
	JSX = 'jsx',
	TSX = 'tsx',
	CSS = 'css',
	HTML = 'html',
	JSON = 'json',
	XML = 'xml',
	YAML = 'yaml',
	CSV = 'csv',
	LOG = 'log',
	PYTHON = 'python',
	JAVA = 'java',
	CPP = 'cpp',
	PHP = 'php',
	RUBY = 'ruby',
	GO = 'go',
	RUST = 'rust',
	SHELL = 'shell',
	SQL = 'sql',
	R = 'r',
	SCALA = 'scala',
	KOTLIN = 'kotlin',
	SWIFT = 'swift',
	DART = 'dart',
	VUE = 'vue',
	SVELTE = 'svelte'
}

// File extension enums
export enum ImageExtension {
	JPG = '.jpg',
	JPEG = '.jpeg',
	PNG = '.png',
	GIF = '.gif',
	WEBP = '.webp',
	SVG = '.svg'
}

export enum AudioExtension {
	MP3 = '.mp3',
	WAV = '.wav'
}

export enum PdfExtension {
	PDF = '.pdf'
}

export enum TextExtension {
	TXT = '.txt',
	MD = '.md',
	JS = '.js',
	TS = '.ts',
	JSX = '.jsx',
	TSX = '.tsx',
	CSS = '.css',
	HTML = '.html',
	HTM = '.htm',
	JSON = '.json',
	XML = '.xml',
	YAML = '.yaml',
	YML = '.yml',
	CSV = '.csv',
	LOG = '.log',
	PY = '.py',
	JAVA = '.java',
	CPP = '.cpp',
	C = '.c',
	H = '.h',
	PHP = '.php',
	RB = '.rb',
	GO = '.go',
	RS = '.rs',
	SH = '.sh',
	BAT = '.bat',
	SQL = '.sql',
	R = '.r',
	SCALA = '.scala',
	KT = '.kt',
	SWIFT = '.swift',
	DART = '.dart',
	VUE = '.vue',
	SVELTE = '.svelte'
}

// MIME type enums
export enum ImageMimeType {
	JPEG = 'image/jpeg',
	PNG = 'image/png',
	GIF = 'image/gif',
	WEBP = 'image/webp',
	SVG = 'image/svg+xml'
}

export enum AudioMimeType {
	MP3_MPEG = 'audio/mpeg',
	MP3 = 'audio/mp3',
	MP4 = 'audio/mp4',
	WAV = 'audio/wav',
	WEBM = 'audio/webm',
	WEBM_OPUS = 'audio/webm;codecs=opus'
}

export enum PdfMimeType {
	PDF = 'application/pdf'
}

export enum TextMimeType {
	PLAIN = 'text/plain',
	MARKDOWN = 'text/markdown',
	JAVASCRIPT = 'text/javascript',
	JAVASCRIPT_APP = 'application/javascript',
	TYPESCRIPT = 'text/typescript',
	JSX = 'text/jsx',
	TSX = 'text/tsx',
	CSS = 'text/css',
	HTML = 'text/html',
	JSON = 'application/json',
	XML_TEXT = 'text/xml',
	XML_APP = 'application/xml',
	YAML_TEXT = 'text/yaml',
	YAML_APP = 'application/yaml',
	CSV = 'text/csv',
	PYTHON = 'text/x-python',
	JAVA = 'text/x-java-source',
	CPP_SRC = 'text/x-c++src',
	C_SRC = 'text/x-csrc',
	C_HDR = 'text/x-chdr',
	PHP = 'text/x-php',
	RUBY = 'text/x-ruby',
	GO = 'text/x-go',
	RUST = 'text/x-rust',
	SHELL = 'text/x-shellscript',
	BAT = 'application/x-bat',
	SQL = 'text/x-sql',
	R = 'text/x-r',
	SCALA = 'text/x-scala',
	KOTLIN = 'text/x-kotlin',
	SWIFT = 'text/x-swift',
	DART = 'text/x-dart',
	VUE = 'text/x-vue',
	SVELTE = 'text/x-svelte'
}

// File type configuration using enums
export const IMAGE_FILE_TYPES = {
	[ImageFileType.JPEG]: {
		extensions: [ImageExtension.JPG, ImageExtension.JPEG],
		mimeTypes: [ImageMimeType.JPEG],
	},
	[ImageFileType.PNG]: {
		extensions: [ImageExtension.PNG],
		mimeTypes: [ImageMimeType.PNG],
	},
	[ImageFileType.GIF]: {
		extensions: [ImageExtension.GIF],
		mimeTypes: [ImageMimeType.GIF],
	},
	[ImageFileType.WEBP]: {
		extensions: [ImageExtension.WEBP],
		mimeTypes: [ImageMimeType.WEBP],
	},
	[ImageFileType.SVG]: {
		extensions: [ImageExtension.SVG],
		mimeTypes: [ImageMimeType.SVG],
	}
} as const;

export const AUDIO_FILE_TYPES = {
	[AudioFileType.MP3]: {
		extensions: [AudioExtension.MP3],
		mimeTypes: [AudioMimeType.MP3_MPEG, AudioMimeType.MP3],
	},
	[AudioFileType.WAV]: {
		extensions: [AudioExtension.WAV],
		mimeTypes: [AudioMimeType.WAV],
	}
} as const;

export const PDF_FILE_TYPES = {
	[PdfFileType.PDF]: {
		extensions: [PdfExtension.PDF],
		mimeTypes: [PdfMimeType.PDF],
	}
} as const;

export const TEXT_FILE_TYPES = {
	[TextFileType.PLAIN_TEXT]: {
		extensions: [TextExtension.TXT],
		mimeTypes: [TextMimeType.PLAIN],
	},
	[TextFileType.MARKDOWN]: {
		extensions: [TextExtension.MD],
		mimeTypes: [TextMimeType.MARKDOWN],
	},
	[TextFileType.JAVASCRIPT]: {
		extensions: [TextExtension.JS],
		mimeTypes: [TextMimeType.JAVASCRIPT, TextMimeType.JAVASCRIPT_APP],
	},
	[TextFileType.TYPESCRIPT]: {
		extensions: [TextExtension.TS],
		mimeTypes: [TextMimeType.TYPESCRIPT],
	},
	[TextFileType.JSX]: {
		extensions: [TextExtension.JSX],
		mimeTypes: [TextMimeType.JSX],
	},
	[TextFileType.TSX]: {
		extensions: [TextExtension.TSX],
		mimeTypes: [TextMimeType.TSX],
	},
	[TextFileType.CSS]: {
		extensions: [TextExtension.CSS],
		mimeTypes: [TextMimeType.CSS],
	},
	[TextFileType.HTML]: {
		extensions: [TextExtension.HTML, TextExtension.HTM],
		mimeTypes: [TextMimeType.HTML],
	},
	[TextFileType.JSON]: {
		extensions: [TextExtension.JSON],
		mimeTypes: [TextMimeType.JSON],
	},
	[TextFileType.XML]: {
		extensions: [TextExtension.XML],
		mimeTypes: [TextMimeType.XML_TEXT, TextMimeType.XML_APP],
	},
	[TextFileType.YAML]: {
		extensions: [TextExtension.YAML, TextExtension.YML],
		mimeTypes: [TextMimeType.YAML_TEXT, TextMimeType.YAML_APP],
	},
	[TextFileType.CSV]: {
		extensions: [TextExtension.CSV],
		mimeTypes: [TextMimeType.CSV],
	},
	[TextFileType.LOG]: {
		extensions: [TextExtension.LOG],
		mimeTypes: [TextMimeType.PLAIN],
	},
	[TextFileType.PYTHON]: {
		extensions: [TextExtension.PY],
		mimeTypes: [TextMimeType.PYTHON],
	},
	[TextFileType.JAVA]: {
		extensions: [TextExtension.JAVA],
		mimeTypes: [TextMimeType.JAVA],
	},
	[TextFileType.CPP]: {
		extensions: [TextExtension.CPP, TextExtension.C, TextExtension.H],
		mimeTypes: [TextMimeType.CPP_SRC, TextMimeType.C_SRC, TextMimeType.C_HDR],
	},
	[TextFileType.PHP]: {
		extensions: [TextExtension.PHP],
		mimeTypes: [TextMimeType.PHP],
	},
	[TextFileType.RUBY]: {
		extensions: [TextExtension.RB],
		mimeTypes: [TextMimeType.RUBY],
	},
	[TextFileType.GO]: {
		extensions: [TextExtension.GO],
		mimeTypes: [TextMimeType.GO],
	},
	[TextFileType.RUST]: {
		extensions: [TextExtension.RS],
		mimeTypes: [TextMimeType.RUST],
	},
	[TextFileType.SHELL]: {
		extensions: [TextExtension.SH, TextExtension.BAT],
		mimeTypes: [TextMimeType.SHELL, TextMimeType.BAT],
	},
	[TextFileType.SQL]: {
		extensions: [TextExtension.SQL],
		mimeTypes: [TextMimeType.SQL],
	},
	[TextFileType.R]: {
		extensions: [TextExtension.R],
		mimeTypes: [TextMimeType.R],
	},
	[TextFileType.SCALA]: {
		extensions: [TextExtension.SCALA],
		mimeTypes: [TextMimeType.SCALA],
	},
	[TextFileType.KOTLIN]: {
		extensions: [TextExtension.KT],
		mimeTypes: [TextMimeType.KOTLIN],
	},
	[TextFileType.SWIFT]: {
		extensions: [TextExtension.SWIFT],
		mimeTypes: [TextMimeType.SWIFT],
	},
	[TextFileType.DART]: {
		extensions: [TextExtension.DART],
		mimeTypes: [TextMimeType.DART],
	},
	[TextFileType.VUE]: {
		extensions: [TextExtension.VUE],
		mimeTypes: [TextMimeType.VUE],
	},
	[TextFileType.SVELTE]: {
		extensions: [TextExtension.SVELTE],
		mimeTypes: [TextMimeType.SVELTE],
	}
} as const;

// Utility arrays for quick access using enum values
export const ALL_SUPPORTED_EXTENSIONS = [
	...Object.values(ImageExtension),
	...Object.values(AudioExtension),
	...Object.values(PdfExtension),
	...Object.values(TextExtension)
] as const;

export const ALL_SUPPORTED_MIME_TYPES = [
	...Object.values(ImageMimeType),
	...Object.values(AudioMimeType),
	...Object.values(PdfMimeType),
	...Object.values(TextMimeType)
] as const;

// Helper functions for file type detection
export function getFileTypeCategory(mimeType: string): FileTypeCategory | null {
	if (Object.values(IMAGE_FILE_TYPES).some(type => (type.mimeTypes as readonly string[]).includes(mimeType))) {
		return FileTypeCategory.IMAGE;
	}
	if (Object.values(AUDIO_FILE_TYPES).some(type => (type.mimeTypes as readonly string[]).includes(mimeType))) {
		return FileTypeCategory.AUDIO;
	}
	if (Object.values(PDF_FILE_TYPES).some(type => (type.mimeTypes as readonly string[]).includes(mimeType))) {
		return FileTypeCategory.PDF;
	}
	if (Object.values(TEXT_FILE_TYPES).some(type => (type.mimeTypes as readonly string[]).includes(mimeType))) {
		return FileTypeCategory.TEXT;
	}
	return null;
}

export function getFileTypeByExtension(filename: string): string | null {
	const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));
	
	// Check image types
	for (const [key, type] of Object.entries(IMAGE_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.IMAGE}:${key}`;
		}
	}
	
	// Check audio types
	for (const [key, type] of Object.entries(AUDIO_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.AUDIO}:${key}`;
		}
	}
	
	// Check PDF types
	for (const [key, type] of Object.entries(PDF_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.PDF}:${key}`;
		}
	}
	
	// Check text types
	for (const [key, type] of Object.entries(TEXT_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.TEXT}:${key}`;
		}
	}
	
	return null;
}

export function isFileTypeSupported(filename: string, mimeType?: string): boolean {
	if (mimeType && getFileTypeCategory(mimeType)) {
		return true;
	}
	return getFileTypeByExtension(filename) !== null;
}

// Summary statistics using enum-based structure
export const SUPPORTED_FILE_STATS = {
	totalExtensions: ALL_SUPPORTED_EXTENSIONS.length,
	totalMimeTypes: ALL_SUPPORTED_MIME_TYPES.length,
	imageFormats: Object.keys(ImageFileType).length,
	audioFormats: Object.keys(AudioFileType).length,
	pdfFormats: Object.keys(PdfFileType).length,
	textFormats: Object.keys(TextFileType).length,
	categories: Object.keys(FileTypeCategory).length
} as const;
