import Foundation
import TesseraCore

// MARK: - DataLayerHTTPClient

/// A small Swift-side HTTP client for the data layer. Mirrors
/// the Python ``DataLayerClient`` so the in-process
/// ``TesseraImporter`` / ``TesseraExporter`` can talk to a
/// remote data layer (the same instance the macOS app is
/// running) when needed.
///
/// v1 uses this client only for the macOS app's startup
/// configuration check ("is the data layer healthy?"); the
/// actual import / export goes through the Python CLI which
/// talks to the same HTTP API. v2 may use this client for
/// other surfaces (the share sheet's "duplicate" button,
/// for example, calls into the data layer directly).
///
/// The client is a value type because it carries no
/// mutable state beyond the configuration. The network
/// calls are dispatched through ``URLSession.shared``.
public struct DataLayerHTTPClient: Sendable {
    public let baseURL: URL
    public let session: URLSession

    public init(
        baseURL: URL = URL(string: "http://127.0.0.1:8787")!,
        session: URLSession = .shared
    ) {
        self.baseURL = baseURL
        self.session = session
    }

    /// Health check. Returns true when the server responds
    /// 200 on ``/v1/health`` (the endpoint is the
    /// ``ImportExportAPI``'s built-in handler; the Python
    /// side uses the same path).
    public func isHealthy() async -> Bool {
        let url = baseURL.appendingPathComponent("v1/health")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 5
        do {
            let (_, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            return (200..<300).contains(http.statusCode)
        } catch {
            return false
        }
    }

    /// Create a new ``graph_entity``. The Swift side stores
    /// the entity via the local ``TesseraDataLayer``; this
    /// client method is for the case where the entity needs
    /// to be created in a remote data layer (e.g., a worker
    /// process running on another host).
    public func createEntity(
        entityType: String,
        label: String,
        body: String,
        sourceURL: URL? = nil,
        subtype: String? = nil
    ) async throws -> UUID {
        let url = baseURL.appendingPathComponent("v1/entities")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var payload: [String: String] = [
            "entity_type": entityType,
            "label": label,
            "body": body,
        ]
        if let sourceURL { payload["source_url"] = sourceURL.absoluteString }
        if let subtype { payload["subtype"] = subtype }
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw DataLayerHTTPError.unexpectedStatus
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let idStr = json["entity_id"] as? String,
              let id = UUID(uuidString: idStr) else {
            throw DataLayerHTTPError.malformedResponse
        }
        return id
    }

    /// Get the body of an entity.
    public func getEntityBody(id: UUID) async throws -> String? {
        let url = baseURL.appendingPathComponent("v1/entities/\(id.uuidString)")
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        let (data, response) = try await session.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode == 404 {
            return nil
        }
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw DataLayerHTTPError.unexpectedStatus
        }
        return String(data: data, encoding: .utf8)
    }

    /// Append a receipt to the entity's chain.
    public func appendReceipt(
        entityID: UUID,
        receiptType: String,
        payload: [String: Any],
        signature: Data? = nil
    ) async throws -> UUID {
        let url = baseURL.appendingPathComponent("v1/receipts")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = [
            "entity_id": entityID.uuidString,
            "receipt_type": receiptType,
            "payload": payload,
        ]
        if let signature {
            body["signature"] = signature.base64EncodedString()
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            throw DataLayerHTTPError.unexpectedStatus
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let idStr = json["receipt_id"] as? String,
              let id = UUID(uuidString: idStr) else {
            throw DataLayerHTTPError.malformedResponse
        }
        return id
    }
}

public enum DataLayerHTTPError: Error, LocalizedError {
    case unexpectedStatus
    case malformedResponse
    public var errorDescription: String? {
        switch self {
        case .unexpectedStatus: return "Data layer HTTP: unexpected status code"
        case .malformedResponse: return "Data layer HTTP: malformed response body"
        }
    }
}
