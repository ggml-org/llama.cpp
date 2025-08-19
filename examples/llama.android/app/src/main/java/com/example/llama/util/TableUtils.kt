package com.example.llama.util


/**
 * A basic
 */
data class MarkdownTableData(
    val headers: List<String>,
    val rows: List<List<String>>
) {
    val columnCount: Int get() = headers.size
    val rowCount: Int get() = rows.size
}

/**
 * Formats llama-bench's markdown output into structured [MarkdownTableData]
 */
fun parseMarkdownTableFiltered(
    markdown: String,
    keepColumns: Set<String>
): MarkdownTableData {
    val lines = markdown.trim().lines().filter { it.startsWith("|") }
    if (lines.size < 2) return MarkdownTableData(emptyList(), emptyList())

    val rawHeaders = lines[0].split("|").map { it.trim() }.filter { it.isNotEmpty() }
    val keepIndices = rawHeaders.mapIndexedNotNull { index, name ->
        if (name in keepColumns) index else null
    }

    val headers = keepIndices.map { rawHeaders[it] }

    val rows = lines.drop(2).map { line ->
        val cells = line.split("|").map { it.trim() }.filter { it.isNotEmpty() }
        keepIndices.map { cells.getOrElse(it) { "" } }
    }

    return MarkdownTableData(headers, rows)
}
