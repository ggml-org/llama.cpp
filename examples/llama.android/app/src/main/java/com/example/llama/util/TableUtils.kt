package com.example.llama.util


/**
 * A basic table data holder separating rows and columns
 */
data class TableData(
    val headers: List<String>,
    val rows: List<List<String>>
) {
    val columnCount: Int get() = headers.size
    val rowCount: Int get() = rows.size

    /**
     * Generate a copy of the original table with only the [keep] columns
     */
    fun filterColumns(keep: Set<String>): TableData =
        headers.mapIndexedNotNull { index, name ->
            if (name in keep) index else null
        }.let { keepIndices ->
            val newHeaders = keepIndices.map { headers[it] }
            val newRows = rows.map { row -> keepIndices.map { row.getOrElse(it) { "" } } }
            TableData(newHeaders, newRows)
        }

    /**
     * Obtain the data in the specified column
     */
    fun getColumn(name: String): List<String> {
        val index = headers.indexOf(name)
        if (index == -1) return emptyList()
        return rows.mapNotNull { it.getOrNull(index) }
    }
}

/**
 * Formats llama-bench's markdown output into structured [MarkdownTableData]
 */
fun parseMarkdownTable(markdown: String): TableData {
    val lines = markdown.trim().lines().filter { it.startsWith("|") }
    if (lines.size < 2) return TableData(emptyList(), emptyList())

    val headers = lines[0].split("|").map { it.trim() }.filter { it.isNotEmpty() }
    val rows = lines.drop(2).map { line ->
        line.split("|").map { it.trim() }.filter { it.isNotEmpty() }
    }

    return TableData(headers, rows)
}
