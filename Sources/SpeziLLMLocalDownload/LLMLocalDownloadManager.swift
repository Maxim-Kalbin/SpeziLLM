//
// This source file is part of the Stanford Spezi open source project
//
// SPDX-FileCopyrightText: 2022 Stanford University and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import Hub
import Observation
import SpeziLLMLocal
import SpeziViews


/// Manages the download and storage of Large Language Models (LLM) to the local device.
///
/// One configures the ``LLMLocalDownloadManager`` via the ``LLMLocalDownloadManager/init(model:)`` initializer,
/// passing a download `URL` as well as a storage `URL` to the ``LLMLocalDownloadManager``.
/// The download of a model is started via ``LLMLocalDownloadManager/startDownload()`` and can be cancelled (early) via ``LLMLocalDownloadManager/cancelDownload()``.
///
/// The current state of the ``LLMLocalDownloadManager`` is exposed via the ``LLMLocalDownloadManager/state`` property which
/// is of type ``LLMLocalDownloadManager/DownloadState``, containing states such as ``LLMLocalDownloadManager/DownloadState/downloading(progress:)``
/// which includes the progress of the download or ``LLMLocalDownloadManager/DownloadState/downloaded`` which indicates that the download has finished.
@Observable
public final class LLMLocalDownloadManager: NSObject {
    /// An enum containing all possible states of the ``LLMLocalDownloadManager``.
    public enum DownloadState: Equatable {
        case idle
        case downloading(progress: Progress)
        case downloaded
        case error(any LocalizedError)
        
        
        public static func == (lhs: LLMLocalDownloadManager.DownloadState, rhs: LLMLocalDownloadManager.DownloadState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle): true
            case (.downloading, .downloading): true
            case (.downloaded, .downloaded): true
            case (.error, .error): true
            default: false
            }
        }
    }
    
    /// The `URLSessionDownloadTask` that handles the download of the model.
    @ObservationIgnored private var downloadTask: Task<(), Never>?
    /// Indicates the current state of the ``LLMLocalDownloadManager``.
    @MainActor public var state: DownloadState = .idle
    private let model: LLMLocalModel
    
    @ObservationIgnored public var modelExist: Bool {
        LLMLocalDownloadManager.modelExist(model: model)
    }
    
    /// Initializes a ``LLMLocalDownloadManager`` instance to manage the download of Large Language Model (LLM) files from remote servers.
    ///
    /// - Parameters:
    ///   - model: The Huggingface model ID of the LLM that needs to be downloaded, or a local path for GGUF files.
    public init(model: LLMLocalModel) {
        self.model = model
    }
    
    /// Checks if a model is already downloaded to the local device or if a local GGUF file path exists.
    ///
    /// - Parameter model: The model to check for local existence. For GGUF, `model.hubID` should be the full file path.
    /// - Returns: A Boolean value indicating whether the model exists on the device.
    public static func modelExist(model: LLMLocalModel) -> Bool {
        let modelId = model.hubID

        // Check if the modelId is a path to a local GGUF file
        if modelId.hasSuffix(".gguf") {
            return FileManager.default.fileExists(atPath: modelId)
        }

        // Original logic for Hugging Face repositories and .safetensors files
        let repo = Hub.Repo(id: modelId)
        let url = HubApi.shared.localRepoLocation(repo)
        let modelFileExtension = ".safetensors"
        
        do {
            let contents = try FileManager.default.contentsOfDirectory(atPath: url.path())
            return contents.contains { $0.hasSuffix(modelFileExtension) }
        } catch {
            // If directory doesn't exist or other error, model is not considered downloaded this way.
            return false
        }
    }
    
    /// Starts a `URLSessionDownloadTask` to download the specified model.
    /// This method will not attempt to download if `model.hubID` points to an existing local GGUF file.
    @MainActor
    public func startDownload() async {
        if modelExist { // This check now also handles local GGUF files
            state = .downloaded
            return
        }
        
        // If it's a GGUF path but doesn't exist, we can't download it with current Hub logic.
        // This part of the logic might need to be adjusted if GGUF download from Hub is desired.
        // For now, we assume GGUF files are manually placed or this download path is for HuggingFace models.
        if model.hubID.hasSuffix(".gguf") && !modelExist {
            state = .error(
                AnyLocalizedError(
                    error: NSError(domain: "LLMLocalDownloadManager", code: -1, userInfo: [NSLocalizedDescriptionKey: "Local GGUF file specified but not found at path: \(model.hubID)"]),
                    defaultErrorDescription: LocalizedStringResource("LLM_GGUF_NOT_FOUND_ERROR", bundle: .atURL(from: .module))
                )
            )
            return
        }

        await cancelDownload()
        downloadTask = Task(priority: .userInitiated) {
            do {
                try await downloadWithHub()
                state = .downloaded
            } catch {
                state = .error(
                    AnyLocalizedError(
                        error: error,
                        defaultErrorDescription: LocalizedStringResource(
                            "LLM_DOWNLOAD_FAILED_ERROR",
                            bundle: .atURL(from: .module)
                        )
                    )
                )
            }
        }
    }
    
    /// Cancels the download of a specified model via a `URLSessionDownloadTask`.
    @MainActor
    public func cancelDownload() async {
        downloadTask?.cancel()
        state = .idle
    }

    private func downloadWithHub() async throws {
        // Sadly, we need this workaround to make the Swift compiler (strict concurrency checking) happy
        @MainActor
        func mutate(progress: Progress) {
              self.state = .downloading(progress: progress)
        }

        let repo = Hub.Repo(id: model.hubID)
        // For typical HuggingFace model downloads, these are the common files.
        // This list might need adjustment if downloading GGUF from Hub repos (though GGUF usually isn't split like safetensors).
        let modelFiles = ["*.safetensors", "config.json", "*.gguf"] // Added *.gguf if Hub.snapshot supports it.
        
        try await HubApi.shared.snapshot(from: repo, matching: modelFiles) { progress in
            Task { @MainActor [mutate] in
                mutate(progress)
            }
        }
    }
}
