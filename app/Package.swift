// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "MacApp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "MacApp", targets: ["MacApp"]),
    ],
    targets: [
        // Pure, dependency-free protocol code (audio.sock framing, NDJSON). Unit-tested.
        .target(name: "MacProtocol"),
        .executableTarget(
            name: "MacApp",
            dependencies: ["MacProtocol"],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("Carbon"),
                .linkedFramework("ServiceManagement"),
                .linkedFramework("UserNotifications"),
                .linkedFramework("ApplicationServices"),
            ]
        ),
        .testTarget(name: "MacProtocolTests", dependencies: ["MacProtocol"]),
    ]
)
