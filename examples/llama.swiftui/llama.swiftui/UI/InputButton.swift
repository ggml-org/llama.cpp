import SwiftUI

struct InputButton: View {
    @ObservedObject var llamaState: LlamaState
    @State private var inputLink: String = ""
    @State private var status: String = "download"
    @State private var filename: String = ""

    @State private var downloadTask: URLSessionDownloadTask?
    @State private var progress = 0.0
    @State private var observation: NSKeyValueObservation?

    private static func extractModelInfo(from link: String) -> (modelName: String, filename: String)? {
        guard let url = URL(string: link),
              let lastPathComponent = url.lastPathComponent.components(separatedBy: ".").first,
              let modelName = lastPathComponent.components(separatedBy: "-").dropLast().joined(separator: "-").removingPercentEncoding,
              let filename = lastPathComponent.removingPercentEncoding else {
            return nil
        }

        return (modelName, filename)
    }

    private static func getFileURL(filename: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(filename)
    }

    private func download() {
        guard let extractedInfo = InputButton.extractModelInfo(from: inputLink) else {
            // Handle invalid link or extraction failure
            return
        }

        let (modelName, filename) = extractedInfo
        self.filename = filename  // Set the state variable

        status = "downloading"
        print("Downloading model \(modelName) from \(inputLink)")
        guard let url = URL(string: inputLink) else { return }
        let fileURL = InputButton.getFileURL(filename: filename)

        downloadTask = URLSession.shared.downloadTask(with: url) { temporaryURL, response, error in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }

            guard let response = response as? HTTPURLResponse, (200...299).contains(response.statusCode) else {
                print("Server error!")
                return
            }

            do {
                if let temporaryURL = temporaryURL {
                    try FileManager.default.copyItem(at: temporaryURL, to: fileURL)
                    print("Writing to \(filename) completed")

                    llamaState.cacheCleared = false

                    let model = Model(name: modelName, url: self.inputLink, filename: filename, status: "downloaded")
                    llamaState.downloadedModels.append(model)
                    status = "downloaded"
                }
            } catch let err {
                print("Error: \(err.localizedDescription)")
            }
        }

        observation = downloadTask?.progress.observe(\.fractionCompleted) { progress, _ in
            self.progress = progress.fractionCompleted
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
