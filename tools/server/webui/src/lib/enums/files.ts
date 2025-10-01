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
export enum FileTypeImage {
	JPEG = 'jpeg',
	PNG = 'png',
	GIF = 'gif',
	WEBP = 'webp',
	SVG = 'svg'
}

export enum FileTypeAudio {
	MP3 = 'mp3',
	WAV = 'wav',
	WEBM = 'webm'
}

export enum FileTypePdf {
	PDF = 'pdf'
}

export enum FileTypeText {
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
	SVELTE = 'svelte',
	LATEX = 'latex',
	BIBTEX = 'bibtex'
}

// File extension enums
export enum FileExtensionImage {
	JPG = '.jpg',
	JPEG = '.jpeg',
	PNG = '.png',
	GIF = '.gif',
	WEBP = '.webp',
	SVG = '.svg'
}

export enum FileExtensionAudio {
	MP3 = '.mp3',
	WAV = '.wav'
}

export enum FileExtensionPdf {
	PDF = '.pdf'
}

export enum FileExtensionText {
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
	SVELTE = '.svelte',
	TEX = '.tex',
	BIB = '.bib'
}

// MIME type enums
export enum MimeTypeApplication {
	PDF = 'application/pdf'
}

export enum MimeTypeAudio {
	MP3_MPEG = 'audio/mpeg',
	MP3 = 'audio/mp3',
	MP4 = 'audio/mp4',
	WAV = 'audio/wav',
	WEBM = 'audio/webm',
	WEBM_OPUS = 'audio/webm;codecs=opus'
}

export enum MimeTypeImage {
	JPEG = 'image/jpeg',
	PNG = 'image/png',
	GIF = 'image/gif',
	WEBP = 'image/webp',
	SVG = 'image/svg+xml'
}

export enum MimeTypeText {
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
	SVELTE = 'text/x-svelte',
	LATEX = 'text/x-tex',
	BIBTEX = 'text/x-bibtex'
}
