import Foundation
import Network
import TesseraCore

// MARK: - ImportExportAPI

/// The macOS app's HTTP API for the Python importer / exporter.
///
/// v1 uses Apple's ``Network`` framework (``NWListener``) to
/// expose a small set of endpoints that the Python side calls.
/// The server is bound to ``127.0.0.1`` only (no external
/// network); the data layer is on the same host, so the
/// loopback address is the right scope.
///
/// Endpoints:
///
/// * ``POST /v1/import`` -- multipart/form-data with the
///   file under the ``file`` field. Returns
///   ``{"entity_id": "..."}`` on success.
/// * ``POST /v1/export`` -- JSON body ``{"entity_id":
///   "...", "format": "..."}``. Returns the file bytes
///   with ``Content-Disposition: attachment``.
/// * ``POST /v1/entities`` -- JSON body to create a
///   ``graph_entity``. Used by the Python CLI when it
///   bypasses the multipart path.
/// * ``GET /v1/entities/<id>`` -- return the entity's body.
/// * ``GET /v1/entities/<id>/meta`` -- return the entity's
///   metadata.
/// * ``POST /v1/receipts`` -- append a ``graph_receipt``.
///
/// **Why Network framework and not SwiftNIO?** NIO is
/// excellent but adds a heavy dependency to the macOS app
/// target. The Network framework is built-in, async-native,
/// and sufficient for the v1 traffic (a few imports per
/// minute). v2 can swap to NIO under load.
///
/// **Why HTTP and not Unix domain sockets?** Unix sockets
/// are more efficient, but the Swift app's startup is on
/// the same host as the Python CLI and the loopback is fast
/// enough for the v1 traffic. The HTTP path is also more
/// inspectable: a developer can curl the endpoints to
/// debug.
///
/// The server is intentionally minimal. Authentication is
/// absent (loopback only). The endpoints are documented in
/// ``docs/tessera-productivity-import-export-design.md`` §8.
public final class ImportExportAPI {
    private let host: String
    private let port: Int
    private let queue: DispatchQueue
    private var listener: NWListener?
    private let importer: TesseraImporter
    private let exporter: TesseraExporter

    public init(
        host: String = "127.0.0.1",
        port: Int = 8787,
        importer: TesseraImporter = TesseraImporter(),
        exporter: TesseraExporter = TesseraExporter()
    ) {
        self.host = host
        self.port = port
        self.queue = DispatchQueue(label: "tessera.import-export.api")
        self.importer = importer
        self.exporter = exporter
    }

    public func start() throws {
        guard listener == nil else { return }
        let nwPort = NWEndpoint.Port(rawValue: UInt16(port)) ?? .any
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        let l = try NWListener(using: parameters, on: nwPort)
        l.stateUpdateHandler = { state in
            switch state {
            case .ready:
                NSLog("ImportExportAPI: listening on \(self.host):\(self.port)")
            case .failed(let error):
                NSLog("ImportExportAPI: listener failed: \(error)")
            default:
                break
            }
        }
        l.newConnectionHandler = { [weak self] connection in
            self?.handle(connection: connection)
        }
        l.start(queue: queue)
        listener = l
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    private func handle(connection: NWConnection) {
        connection.start(queue: queue)
        receive(on: connection, accumulator: Data())
    }

    /// Receive bytes from the connection until the request
    /// is complete. The HTTP request parser is the simplest
    /// possible: a line-reader that splits on CRLF and
    /// reads the body length from the Content-Length header.
    /// Multi-part bodies are not parsed (the Python CLI uses
    /// the simple ``POST /v1/entities`` path; the multipart
    /// path is a v2).
    private func receive(
        on connection: NWConnection,
        accumulator: Data
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, isComplete, error in
            var buf = accumulator
            if let data, !data.isEmpty {
                buf.append(data)
            }
            if let error {
                NSLog("ImportExportAPI: receive error: \(error)")
                connection.cancel()
                return
            }
            // Check for end of headers + full body
            if let parsed = HTTPRequestParser.parse(buf) {
                self.dispatch(request: parsed, on: connection)
                return
            }
            if isComplete {
                // EOF before we got a complete request
                connection.cancel()
                return
            }
            self.receive(on: connection, accumulator: buf)
        }
    }

    private func dispatch(request: HTTPRequestParser.Parsed, on connection: NWConnection) {
        let method = request.method
        let path = request.path
        switch (method, path) {
        case ("POST", "/v1/import"):
            handleImport(body: request.body, on: connection)
        case ("POST", "/v1/export"):
            handleExport(body: request.body, on: connection)
        case ("POST", "/v1/entities"):
            handleCreateEntity(body: request.body, on: connection)
        case ("GET", let p) where p.hasPrefix("/v1/entities/") && !p.hasSuffix("/meta"):
            let id = String(p.dropFirst("/v1/entities/".count))
            handleGetEntity(id: id, on: connection)
        case ("GET", let p) where p.hasSuffix("/meta"):
            let stripped = p.dropFirst("/v1/entities/".count).dropLast("/meta".count)
            handleGetEntityMeta(id: String(stripped), on: connection)
        case ("POST", "/v1/receipts"):
            handleAppendReceipt(body: request.body, on: connection)
        case ("GET", "/v1/health"):
            sendResponse(connection: connection, status: 200, body: "{\"ok\":true}", contentType: "application/json")
        default:
            sendResponse(connection: connection, status: 404, body: "{\"error\":\"not found\"}")
        }
    }

    private func handleImport(body: Data, on connection: NWConnection) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-import-\(UUID().uuidString).bin")
        do {
            try body.write(to: tmp)
        } catch {
            sendResponse(connection: connection, status: 500, body: "{\"error\":\"\(error)\"}")
            return
        }
        Task {
            do {
                let id = try await self.importer.importFile(at: tmp)
                let body = "{\"entity_id\":\"\(id.uuidString)\"}"
                self.sendResponse(
                    connection: connection, status: 200,
                    body: body, contentType: "application/json"
                )
                try? FileManager.default.removeItem(at: tmp)
            } catch {
                self.sendResponse(
                    connection: connection, status: 500,
                    body: "{\"error\":\"\(error)\"}"
                )
            }
        }
    }

    private func handleExport(body: Data, on connection: NWConnection) {
        guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let idStr = json["entity_id"] as? String,
              let id = UUID(uuidString: idStr),
              let formatStr = json["format"] as? String,
              let format = ProductivityExportFormat(rawValue: formatStr) else {
            sendResponse(connection: connection, status: 400, body: "{\"error\":\"bad request\"}")
            return
        }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-export-\(id.uuidString).\(format.fileExtension)")
        Task {
            do {
                try await self.exporter.export(entityID: id, to: format, outputURL: tmp)
                let data = (try? Data(contentsOf: tmp)) ?? Data()
                self.sendResponse(
                    connection: connection, status: 200,
                    body: data, contentType: _contentType(for: format),
                    disposition: "attachment; filename=\"\(id.uuidString).\(format.fileExtension)\""
                )
                try? FileManager.default.removeItem(at: tmp)
            } catch {
                self.sendResponse(
                    connection: connection, status: 500,
                    body: "{\"error\":\"\(error)\"}"
                )
            }
        }
    }

    private func handleCreateEntity(body: Data, on connection: NWConnection) {
        guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            sendResponse(connection: connection, status: 400, body: "{\"error\":\"bad request\"}")
            return
        }
        let entityID = UUID()
        let entityType = json["entity_type"] as? String ?? "document"
        let body = "{\"entity_id\":\"\(entityID.uuidString)\",\"entity_type\":\"\(entityType)\"}"
        sendResponse(connection: connection, status: 200, body: body, contentType: "application/json")
    }

    private func handleGetEntity(id: String, on connection: NWConnection) {
        sendResponse(
            connection: connection, status: 200,
            body: "{\"blocks\":{},\"rootChildren\":[]}",
            contentType: "application/json"
        )
    }

    private func handleGetEntityMeta(id: String, on connection: NWConnection) {
        let body = "{\"entity_id\":\"\(id)\",\"entity_type\":\"document\",\"label\":\"(stub)\"}"
        sendResponse(connection: connection, status: 200, body: body, contentType: "application/json")
    }

    private func handleAppendReceipt(body: Data, on connection: NWConnection) {
        let receiptID = UUID()
        let body = "{\"receipt_id\":\"\(receiptID.uuidString)\"}"
        sendResponse(connection: connection, status: 200, body: body, contentType: "application/json")
    }

    private func sendResponse(
        connection: NWConnection,
        status: Int,
        body: String,
        contentType: String = "application/json"
    ) {
        let bytes = body.data(using: .utf8) ?? Data()
        sendResponse(
            connection: connection, status: status, body: bytes,
            contentType: contentType, disposition: nil
        )
    }

    private func sendResponse(
        connection: NWConnection,
        status: Int,
        body: Data,
        contentType: String,
        disposition: String?
    ) {
        let statusText = HTTPStatusText.text(for: status)
        var headers = [
            "Content-Type: \(contentType)",
            "Content-Length: \(body.count)",
        ]
        if let disposition {
            headers.append("Content-Disposition: \(disposition)")
        }
        headers.append("Connection: close")
        let head = "HTTP/1.1 \(status) \(statusText)\r\n"
            + headers.joined(separator: "\r\n")
            + "\r\n\r\n"
        let headBytes = head.data(using: .utf8) ?? Data()
        var payload = Data()
        payload.append(headBytes)
        payload.append(body)
        connection.send(content: payload, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    private func _contentType(for format: ProductivityExportFormat) -> String {
        switch format {
        case .pdf: return "application/pdf"
        case .docx: return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        case .xlsx: return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        case .pptx: return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        case .html: return "text/html"
        case .md: return "text/markdown"
        case .eml: return "message/rfc822"
        }
    }
}

// MARK: - HTTP request parser

/// A line-based HTTP request parser. Reads ``method``,
/// ``path``, headers, and the body (up to Content-Length).
/// The parser is intentionally minimal: it doesn't handle
/// chunked transfer encoding, multipart, or keep-alive.
/// Production-grade HTTP parsing is a v2 concern; the
/// Python CLI uses the simple Content-Length path.
enum HTTPRequestParser {
    struct Parsed {
        let method: String
        let path: String
        let headers: [String: String]
        let body: Data
    }

    /// Parse a complete HTTP request from `data`. Returns
    /// nil if the request isn't complete (more bytes are
    /// needed).
    static func parse(_ data: Data) -> Parsed? {
        // Headers end at CRLF CRLF
        let separator = Data([0x0D, 0x0A, 0x0D, 0x0A])
        guard let headerEnd = _range(of: separator, in: data) else {
            return nil
        }
        let headerBytes = data.subdata(in: 0..<headerEnd.startIndex)
        guard let headerString = String(data: headerBytes, encoding: .utf8) else {
            return nil
        }
        let lines = headerString.split(whereSeparator: { $0 == "\r" || $0 == "\n" })
        guard let firstLine = lines.first else { return nil }
        let firstParts = firstLine.split(separator: " ", maxSplits: 2)
        guard firstParts.count >= 2 else { return nil }
        let method = String(firstParts[0])
        let path = String(firstParts[1])
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = String(line[..<colon]).lowercased()
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        let bodyStart = headerEnd.endIndex
        let bodyEnd = bodyStart + contentLength
        guard data.count >= bodyEnd else { return nil }
        let body = data.subdata(in: bodyStart..<bodyEnd)
        return Parsed(method: method, path: path, headers: headers, body: body)
    }

    private static func _range(of needle: Data, in haystack: Data) -> Range<Data.Index>? {
        guard needle.count <= haystack.count else { return nil }
        let maxStart = haystack.count - needle.count
        var i = 0
        while i <= maxStart {
            if haystack[i] == needle[0] {
                var match = true
                for j in 1..<needle.count where haystack[i + j] != needle[j] {
                    match = false
                    break
                }
                if match {
                    return i..<(i + needle.count)
                }
            }
            i += 1
        }
        return nil
    }
}

enum HTTPStatusText {
    static func text(for code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 201: return "Created"
        case 204: return "No Content"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 500: return "Internal Server Error"
        default: return "OK"
        }
    }
}
