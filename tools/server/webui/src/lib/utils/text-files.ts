import { TEXT_FILE_EXTENSIONS } from "$lib/constants/text-file-extensions";

export function isTextFileByName(filename: string): boolean {
    return TEXT_FILE_EXTENSIONS.some((ext) => filename.toLowerCase().endsWith(ext));
}

export async function readFileAsText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            if (event.target?.result) {
                resolve(event.target.result as string);
            } else {
                reject(new Error('Failed to read file'));
            }
        };
        reader.onerror = () => reject(new Error('File reading error'));
        reader.readAsText(file);
    });
}

export function isLikelyTextFile(content: string): boolean {
    if (!content) return true;

    const sample = content.substring(0, 1000);

    let suspiciousCount = 0;
    let nullCount = 0;

    for (let i = 0; i < sample.length; i++) {
        const charCode = sample.charCodeAt(i);

        // Count null bytes
        if (charCode === 0) {
            nullCount++;
            suspiciousCount++;
            continue;
        }

        // Count suspicious control characters (excluding common ones like tab, newline, carriage return)
        if (charCode < 32 && charCode !== 9 && charCode !== 10 && charCode !== 13) {
            suspiciousCount++;
        }

        // Count replacement characters (indicates encoding issues)
        if (charCode === 0xfffd) {
            suspiciousCount++;
        }
    }

    // Reject if too many null bytes or suspicious characters
    if (nullCount > 2) return false;
    if (suspiciousCount / sample.length > 0.1) return false;

    return true;
}
