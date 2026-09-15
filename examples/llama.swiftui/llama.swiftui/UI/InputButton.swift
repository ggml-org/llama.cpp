import SwiftUI

struct InputButton: View {
    @ObservedObject var llamaState: LlamaState
    @State private var inputLink: String = ""
    @State private var status: String = "download"
    @State private var filename: String = ""

    @State private var downloadTask: URLSessionDownloadTask?
    @State private var progress = 0.0
    @State private var observation: NSKeyValueObservation?

    // URLSession that follows redirects
    private let urlSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300 // 5 minutes
        config.timeoutIntervalForResource = 3600 // 1 hour
        return URLSession(configuration: config)
    }()

    private static func extractModelInfo(from link: String) -> (modelName: String, filename: String)? {
        guard let url = URL(string: link) else {
            return nil
        }

        // Get the full filename (e.g., "Meta-Llama-3.1-8B-Instruct-Q4_0.gguf")
        let filename = url.lastPathComponent

        // Extract model name by removing .gguf extension
        let modelName = filename.replacingOccurrences(of: ".gguf", with: "")

        // Validate it's a GGUF file
        guard filename.hasSuffix(".gguf") else {
            return nil
        }

        return (modelName, filename)
    }

    private static func getFileURL(filename: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
    }

    private func download() {
        // Trim whitespace from URL
        let trimmedLink = inputLink.trimmingCharacters(in: .whitespacesAndNewlines)

        // Debug: Show what URL we're working with
        llamaState.messageLog += "🔍 Starting download process...\n"
        llamaState.messageLog += "Input URL: \(trimmedLink)\n"

        guard let extractedInfo = InputButton.extractModelInfo(from: trimmedLink) else {
            // Handle invalid link or extraction failure
            llamaState.messageLog += "❌ Invalid download link - failed to extract model info\n"
            llamaState.messageLog += "URL must end with .gguf\n"
            status = "download"
            return
        }

        let (modelName, filename) = extractedInfo
        self.filename = filename  // Set the state variable

        llamaState.messageLog += "✓ Parsed filename: \(filename)\n"
        llamaState.messageLog += "✓ Model name: \(modelName)\n"

        let fileURL = InputButton.getFileURL(filename: filename)

        // Delete existing file if it exists (from failed download)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            llamaState.messageLog += "⚠️ Removing existing incomplete file...\n"
            try? FileManager.default.removeItem(at: fileURL)
        }

        status = "downloading"

        llamaState.messageLog += "📥 Downloading \(modelName)...\n"
        llamaState.messageLog += "This may take 5-10 minutes for large models...\n"

        print("Downloading model \(modelName) from \(trimmedLink)")
        guard let url = URL(string: trimmedLink) else {
            llamaState.messageLog += "❌ Invalid URL format\n"
            status = "download"
            return
        }

        llamaState.messageLog += "✓ Will save to: \(fileURL.path)\n"
        llamaState.messageLog += "✓ Starting download task...\n"

        downloadTask = urlSession.downloadTask(with: url) { temporaryURL, response, error in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.llamaState.messageLog += "❌ Download failed: \(error.localizedDescription)\n"
                    self.status = "download"
                }
                return
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                DispatchQueue.main.async {
                    self.llamaState.messageLog += "❌ Invalid response type\n"
                    self.status = "download"
                }
                return
            }

            DispatchQueue.main.async {
                self.llamaState.messageLog += "📡 Server response code: \(httpResponse.statusCode)\n"
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                print("Server error: \(httpResponse.statusCode)")
                DispatchQueue.main.async {
                    self.llamaState.messageLog += "❌ Server error: HTTP \(httpResponse.statusCode)\n"
                    if let url = httpResponse.url {
                        self.llamaState.messageLog += "Response URL: \(url)\n"
                    }
                    self.status = "download"
                }
                return
            }

            do {
                if let temporaryURL = temporaryURL {
                    try FileManager.default.copyItem(at: temporaryURL, to: fileURL)
                    print("Writing to \(filename) completed")

                    DispatchQueue.main.async {
                        self.llamaState.cacheCleared = false

                        let model = Model(name: modelName, url: trimmedLink, filename: filename, status: "downloaded")
                        self.llamaState.downloadedModels.append(model)
                        self.status = "downloaded"

                        self.llamaState.messageLog += "✅ Download complete: \(modelName)\n"
                        self.llamaState.messageLog += "File saved: \(filename)\n"
                    }
                }
            } catch let err {
                print("Error: \(err.localizedDescription)")
                DispatchQueue.main.async {
                    self.llamaState.messageLog += "❌ File save error: \(err.localizedDescription)\n"
                    self.status = "download"
                }
            }
        }

        observation = downloadTask?.progress.observe(\.fractionCompleted) { observedProgress, _ in
            DispatchQueue.main.async {
                self.progress = observedProgress.fractionCompleted

                // Log progress at 25%, 50%, 75%, 100%
                let percentage = Int(observedProgress.fractionCompleted * 100)
                if percentage % 25 == 0 && percentage > 0 {
                    self.llamaState.messageLog += "📊 Download progress: \(percentage)%\n"
                }
            }
        }

        downloadTask?.resume()
    }

    var body: some View {
        VStack {
            HStack {
                TextField("Paste Quantized Download Link", text: $inputLink)
                    .textFieldStyle(RoundedBorderTextFieldStyle())

                Button(action: {
                    downloadTask?.cancel()
                    status = "download"
                }) {
                    Text("Cancel")
                }
            }

            if status == "download" {
                Button(action: download) {
                    Text("Download Custom Model")
                }
            } else if status == "downloading" {
                Button(action: {
                    downloadTask?.cancel()
                    status = "download"
                }) {
                    Text("Downloading \(Int(progress * 100))%")
                }
            } else if status == "downloaded" {
                Button(action: {
                    let fileURL = InputButton.getFileURL(filename: self.filename)
                    if !FileManager.default.fileExists(atPath: fileURL.path) {
                        download()
                        return
                    }
                    do {
                        try llamaState.loadModel(modelUrl: fileURL)
                    } catch let err {
                        print("Error: \(err.localizedDescription)")
                    }
                }) {
                    Text("Load Custom Model")
                }
            } else {
                Text("Unknown status")
            }
        }
        .onDisappear() {
            downloadTask?.cancel()
        }
        .onChange(of: llamaState.cacheCleared) { newValue in
            if newValue {
                downloadTask?.cancel()
                let fileURL = InputButton.getFileURL(filename: self.filename)
                status = FileManager.default.fileExists(atPath: fileURL.path) ? "downloaded" : "download"
            }
        }
    }
}
