// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "TesseraStudio",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .executable(name: "TesseraStudio", targets: ["TesseraStudio"]),
    ],
    targets: [
        .executableTarget(
            name: "TesseraStudio",
            path: "Sources/TesseraStudio"
        ),
    ]
)
