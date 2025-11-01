/**
 * Comprehensive dictionary of all supported file types in webui
 * Organized by category with TypeScript enums for better type safety
 */

import {
	FileExtensionAudio,
	FileExtensionImage,
	FileExtensionPdf,
	FileExtensionText,
	FileTypeAudio,
	FileTypeImage,
	FileTypePdf,
	FileTypeText,
	MimeTypeAudio,
	MimeTypeImage,
	MimeTypeApplication,
	MimeTypeText
} from '$lib/enums/files';

// File type configuration using enums
export const AUDIO_FILE_TYPES = {
	[FileTypeAudio.MP3]: {
		extensions: [FileExtensionAudio.MP3],
		mimeTypes: [MimeTypeAudio.MP3_MPEG, MimeTypeAudio.MP3]
	},
	[FileTypeAudio.WAV]: {
		extensions: [FileExtensionAudio.WAV],
		mimeTypes: [MimeTypeAudio.WAV]
	}
} as const;

export const IMAGE_FILE_TYPES = {
	[FileTypeImage.JPEG]: {
		extensions: [FileExtensionImage.JPG, FileExtensionImage.JPEG],
		mimeTypes: [MimeTypeImage.JPEG]
	},
	[FileTypeImage.PNG]: {
		extensions: [FileExtensionImage.PNG],
		mimeTypes: [MimeTypeImage.PNG]
	},
	[FileTypeImage.GIF]: {
		extensions: [FileExtensionImage.GIF],
		mimeTypes: [MimeTypeImage.GIF]
	},
	[FileTypeImage.WEBP]: {
		extensions: [FileExtensionImage.WEBP],
		mimeTypes: [MimeTypeImage.WEBP]
	},
	[FileTypeImage.SVG]: {
		extensions: [FileExtensionImage.SVG],
		mimeTypes: [MimeTypeImage.SVG]
	}
} as const;

export const PDF_FILE_TYPES = {
	[FileTypePdf.PDF]: {
		extensions: [FileExtensionPdf.PDF],
		mimeTypes: [MimeTypeApplication.PDF]
	}
} as const;

export const TEXT_FILE_TYPES = {
	[FileTypeText.PLAIN_TEXT]: {
		extensions: [FileExtensionText.TXT],
		mimeTypes: [MimeTypeText.PLAIN]
	},
	[FileTypeText.MARKDOWN]: {
		extensions: [FileExtensionText.MD],
		mimeTypes: [MimeTypeText.MARKDOWN]
	},
	[FileTypeText.ASCIIDOC]: {
		extensions: [FileExtensionText.ADOC],
		mimeTypes: [MimeTypeText.ASCIIDOC]
	},
	[FileTypeText.JAVASCRIPT]: {
		extensions: [FileExtensionText.JS],
		mimeTypes: [MimeTypeText.JAVASCRIPT, MimeTypeText.JAVASCRIPT_APP]
	},
	[FileTypeText.TYPESCRIPT]: {
		extensions: [FileExtensionText.TS],
		mimeTypes: [MimeTypeText.TYPESCRIPT]
	},
	[FileTypeText.JSX]: {
		extensions: [FileExtensionText.JSX],
		mimeTypes: [MimeTypeText.JSX]
	},
	[FileTypeText.TSX]: {
		extensions: [FileExtensionText.TSX],
		mimeTypes: [MimeTypeText.TSX]
	},
	[FileTypeText.CSS]: {
		extensions: [FileExtensionText.CSS],
		mimeTypes: [MimeTypeText.CSS]
	},
	[FileTypeText.HTML]: {
		extensions: [FileExtensionText.HTML, FileExtensionText.HTM],
		mimeTypes: [MimeTypeText.HTML]
	},
	[FileTypeText.JSON]: {
		extensions: [FileExtensionText.JSON],
		mimeTypes: [MimeTypeText.JSON]
	},
	[FileTypeText.XML]: {
		extensions: [FileExtensionText.XML],
		mimeTypes: [MimeTypeText.XML_TEXT, MimeTypeText.XML_APP]
	},
	[FileTypeText.YAML]: {
		extensions: [FileExtensionText.YAML, FileExtensionText.YML],
		mimeTypes: [MimeTypeText.YAML_TEXT, MimeTypeText.YAML_APP]
	},
	[FileTypeText.CSV]: {
		extensions: [FileExtensionText.CSV],
		mimeTypes: [MimeTypeText.CSV]
	},
	[FileTypeText.LOG]: {
		extensions: [FileExtensionText.LOG],
		mimeTypes: [MimeTypeText.PLAIN]
	},
	[FileTypeText.PYTHON]: {
		extensions: [FileExtensionText.PY],
		mimeTypes: [MimeTypeText.PYTHON]
	},
	[FileTypeText.JAVA]: {
		extensions: [FileExtensionText.JAVA],
		mimeTypes: [MimeTypeText.JAVA]
	},
	[FileTypeText.CPP]: {
		extensions: [FileExtensionText.CPP, FileExtensionText.C, FileExtensionText.H],
		mimeTypes: [MimeTypeText.CPP_SRC, MimeTypeText.C_SRC, MimeTypeText.C_HDR]
	},
	[FileTypeText.PHP]: {
		extensions: [FileExtensionText.PHP],
		mimeTypes: [MimeTypeText.PHP]
	},
	[FileTypeText.RUBY]: {
		extensions: [FileExtensionText.RB],
		mimeTypes: [MimeTypeText.RUBY]
	},
	[FileTypeText.GO]: {
		extensions: [FileExtensionText.GO],
		mimeTypes: [MimeTypeText.GO]
	},
	[FileTypeText.RUST]: {
		extensions: [FileExtensionText.RS],
		mimeTypes: [MimeTypeText.RUST]
	},
	[FileTypeText.SHELL]: {
		extensions: [FileExtensionText.SH, FileExtensionText.BAT],
		mimeTypes: [MimeTypeText.SHELL, MimeTypeText.BAT]
	},
	[FileTypeText.SQL]: {
		extensions: [FileExtensionText.SQL],
		mimeTypes: [MimeTypeText.SQL]
	},
	[FileTypeText.R]: {
		extensions: [FileExtensionText.R],
		mimeTypes: [MimeTypeText.R]
	},
	[FileTypeText.SCALA]: {
		extensions: [FileExtensionText.SCALA],
		mimeTypes: [MimeTypeText.SCALA]
	},
	[FileTypeText.KOTLIN]: {
		extensions: [FileExtensionText.KT],
		mimeTypes: [MimeTypeText.KOTLIN]
	},
	[FileTypeText.SWIFT]: {
		extensions: [FileExtensionText.SWIFT],
		mimeTypes: [MimeTypeText.SWIFT]
	},
	[FileTypeText.DART]: {
		extensions: [FileExtensionText.DART],
		mimeTypes: [MimeTypeText.DART]
	},
	[FileTypeText.VUE]: {
		extensions: [FileExtensionText.VUE],
		mimeTypes: [MimeTypeText.VUE]
	},
	[FileTypeText.SVELTE]: {
		extensions: [FileExtensionText.SVELTE],
		mimeTypes: [MimeTypeText.SVELTE]
	},
	[FileTypeText.LATEX]: {
		extensions: [FileExtensionText.TEX],
		mimeTypes: [MimeTypeText.LATEX]
	},
	[FileTypeText.BIBTEX]: {
		extensions: [FileExtensionText.BIB],
		mimeTypes: [MimeTypeText.BIBTEX]
	}
} as const;
