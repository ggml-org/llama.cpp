//@ts-check
// Helpers to handle markdown content
// by Humans for All
//


export class MarkDown {

    constructor() {
        this.in = {
            pre: false,
            table: false,
        }
        this.md = ""
        this.html = ""
    }

    /**
     * Process a line from markdown content
     * @param {string} line
     */
    process_line(line) {
        let lineA = line.split(' ')
        let lineRStripped = line.trimStart()
        if (this.in.pre) {
            if (lineA[0] == '```') {
                this.in.pre = false
                this.html += "</pre>\n"
            } else {
                this.html += line
            }
            return
        }
        if (line.startsWith ("#")) {
            let hLevel = lineA[0].length
            this.html += `<h${hLevel}>${lineRStripped}</h${hLevel}>\n`
            return
        }
        if (lineA[0] == '```') {
            this.in.pre = true
            this.html += "<pre>\n"
            return
        }
        this.html += `<p>${line}</p>`
    }

    /**
     * Process a bunch of lines in markdown format.
     * @param {string} lines
     */
    process(lines) {
        let linesA = lines.split('\n')
        for(const line of linesA) {
            this.process_line(line)
        }
    }

}
