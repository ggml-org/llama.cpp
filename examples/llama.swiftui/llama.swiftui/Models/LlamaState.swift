import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    private var llamaContext: LlamaContext?
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        do {
            try loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    func loadModel(modelUrl: URL?) throws {
        if let modelUrl {
            messageLog += "Loading model...\n"
            llamaContext = try LlamaContext.create_context(path: modelUrl.path())
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"

            // Assuming that the model is successfully loaded, update the downloaded models
            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
        } else {
            messageLog += "Load a model from the list below\n"
        }
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }


    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: text)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        messageLog += "\(text)"

        Task.detached {
            while await !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                await MainActor.run {
                    self.messageLog += "\(result)"
                }
            }

            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
            let tokens_per_second = Double(await llamaContext.n_len) / t_generation

            await llamaContext.clear()

            await MainActor.run {
                self.messageLog += """
                    \n
                    Done
                    Heat up took \(t_heat)s
                    Generated \(tokens_per_second) t/s\n
                    """
            }
        }
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }
}
