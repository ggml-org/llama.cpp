import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var multiLineText = ""
    @State private var showingHelp = false    // To track if Help Sheet should be shown

    var body: some View {
        NavigationView {
            VStack {
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                }

                // Backend selector
                HStack {
                    Text("Backend:")
                        .font(.system(size: 14))
                    Picker("Backend", selection: $llamaState.selectedBackend) {
                        ForEach(Backend.allCases, id: \.self) { backend in
                            Text(backend.displayName).tag(backend)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }
                .padding(.horizontal)

                TextEditor(text: $multiLineText)
                    .frame(height: 80)
                    .padding()
                    .border(Color.gray, width: 0.5)

                HStack {
                    Button("Send") {
                        sendText()
                    }

                    Button("Bench") {
                        bench()
                    }

                    Button("Clear") {
                        clear()
                    }

                    Button("Copy") {
                        UIPasteboard.general.string = llamaState.messageLog
                    }
                }
                .buttonStyle(.bordered)
                .padding()

                Button("Compare All Backends") {
                    compareAll()
                }
                .buttonStyle(.borderedProminent)
                .padding(.horizontal)

                NavigationLink(destination: DrawerView(llamaState: llamaState)) {
                    Text("View Models")
                }
                .padding()

            }
            .padding()
            .navigationBarTitle("Model Settings", displayMode: .inline)

        }
    }

    func sendText() {
        Task {
            await llamaState.complete(text: multiLineText)
            multiLineText = ""
        }
    }

    func bench() {
        Task {
            await llamaState.bench()
        }
    }

    func clear() {
        Task {
            await llamaState.clear()
        }
    }

    func compareAll() {
        Task {
            let prompt = multiLineText.isEmpty ? "What is the formula for water?" : multiLineText
            let results = await llamaState.compareAllBackends(prompt: prompt)

            // Optionally save results to file
            if !results.isEmpty {
                saveResults(results, prompt: prompt)
            }
        }
    }

    func saveResults(_ results: [InferenceMetrics], prompt: String) {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = formatter.string(from: Date())

        var report = "# Metal-4 Tensor Backend Comparison\n\n"
        report += "**Device:** iPhone 17 Pro Max\n"
        report += "**iOS:** 26.0.1\n"
        report += "**Date:** \(timestamp)\n"
        report += "**Prompt:** \(prompt)\n\n"
        report += "## Results\n\n"
        report += llamaState.formatComparisonTable(results)
        report += "\n\n## Details\n\n"

        for metrics in results {
            report += llamaState.formatMetrics(metrics)
            report += "\n\n"
        }

        // Save to file
        let filename = "backend_comparison_\(timestamp).md"
        let documentsPath = llamaState.getDocumentsDirectory()
        let filePath = documentsPath.appendingPathComponent(filename)

        do {
            try report.write(to: filePath, atomically: true, encoding: .utf8)
            llamaState.messageLog += "\n✅ Results saved to:\n\(filePath.path)\n\n"
            llamaState.messageLog += "📋 Copy the text above and paste to a .md file to share!\n"
        } catch {
            llamaState.messageLog += "\n❌ Error saving results: \(error)\n"
        }

        // Also add the full report to the message log so user can copy it
        llamaState.messageLog += "\n" + String(repeating: "=", count: 50) + "\n"
        llamaState.messageLog += "FULL REPORT (Copy this for GitHub):\n"
        llamaState.messageLog += String(repeating: "=", count: 50) + "\n\n"
        llamaState.messageLog += report
    }

    struct DrawerView: View {

        @ObservedObject var llamaState: LlamaState
        @State private var showingHelp = false
        func delete(at offsets: IndexSet) {
            offsets.forEach { offset in
                let model = llamaState.downloadedModels[offset]
                let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
                do {
                    try FileManager.default.removeItem(at: fileURL)
                } catch {
                    print("Error deleting file: \(error)")
                }
            }

            // Remove models from downloadedModels array
            llamaState.downloadedModels.remove(atOffsets: offsets)
        }

        func getDocumentsDirectory() -> URL {
            let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            return paths[0]
        }
        var body: some View {
            List {
                Section(header: Text("Download Models From Hugging Face")) {
                    HStack {
                        InputButton(llamaState: llamaState)
                    }
                }
                Section(header: Text("Downloaded Models")) {
                    ForEach(llamaState.downloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                    .onDelete(perform: delete)
                }
                Section(header: Text("Default Models")) {
                    ForEach(llamaState.undownloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                }

            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Model Settings", displayMode: .inline).toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") {
                        showingHelp = true
                    }
                }
            }.sheet(isPresented: $showingHelp) {    // Sheet for help modal
                NavigationView {
                    VStack(alignment: .leading) {
                        VStack(alignment: .leading) {
                            Text("1. Make sure the model is in GGUF Format")
                                    .padding()
                            Text("2. Copy the download link of the quantized model")
                                    .padding()
                        }
                        Spacer()
                    }
                    .navigationTitle("Help")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Done") {
                                showingHelp = false
                            }
                        }
                    }
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
