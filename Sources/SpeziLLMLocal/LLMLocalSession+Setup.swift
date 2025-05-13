//
// This source file is part of the Stanford Spezi open source project
//
// SPDX-FileCopyrightText: 2024 Stanford University and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

import Foundation
import Hub
import MLXLLM
import MLXLMCommon


extension LLMLocalSession {
    private func verifyModelDownload() -> Bool {
        let modelId = self.schema.configuration.name // This is the hubID or custom path from LLMLocalModel

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
    
    // swiftlint:disable:next identifier_name
    internal func _setup(continuation: AsyncThrowingStream<String, any Error>.Continuation?) async -> Bool {
#if targetEnvironment(simulator)
        return await _mockSetup(continuation: continuation)
#else
        Self.logger.debug("SpeziLLMLocal: Local LLM is being initialized")
        
        await MainActor.run {
            self.state = .loading
        }
        
        guard verifyModelDownload() else {
            let errorDetail = "Local LLM file for configuration '\(self.schema.configuration.name)' could not be verified. It might not exist or is not a recognized .gguf file or HuggingFace downloaded model."
            Self.logger.error("SpeziLLMLocal: \(errorDetail)")
            if let continuation {
                await finishGenerationWithError(LLMLocalError.modelNotFound, on: continuation)
            }
            // Set state to error if setup fails before generation
            if await self.state == .loading { // only set error if we are still in loading (i.e. setup() was called directly)
                 await MainActor.run {
                    self.state = .error(error: LLMLocalError.modelNotFound)
                 }
            }
            return false
        }
        
        do {
            let modelContainer = try await LLMModelFactory.shared.loadContainer(configuration: self.schema.configuration)
            
            let numParams = await modelContainer.perform { modelContext in
                modelContext.model.numParameters()
            }
            
            await MainActor.run {
                self.modelContainer = modelContainer
                self.numParameters = numParams
                self.state = .ready
            }
        } catch {
            Self.logger.error("SpeziLLMLocal: Failed to load local `modelContainer` for \(self.schema.configuration.name): \(error.localizedDescription)")
            if let continuation {
                 // Pass the more specific error if available
                let llmError = LLMLocalError.generationError // Or a more specific error if identifiable
                continuation.yield(with: .failure(llmError))
                await finishGenerationWithError(llmError, on: continuation)
            }
            // Set state to error if setup fails before generation
            if await self.state == .loading { // only set error if we are still in loading (i.e. setup() was called directly)
                await MainActor.run {
                    self.state = .error(error: LLMLocalError.generationError) // Or a more specific error
                }
            }
            return false
        }
        
        Self.logger.debug("SpeziLLMLocal: Local LLM has finished initializing")
        return true
#endif
    }
    
    private func _mockSetup(continuation: AsyncThrowingStream<String, any Error>.Continuation?) async -> Bool {
        Self.logger.debug("SpeziLLMLocal: Local Mock LLM is being initialized")
        
        await MainActor.run {
            self.state = .loading
        }
        
        try? await Task.sleep(for: .seconds(1))
        
        await MainActor.run {
            self.state = .ready
        }
        
        Self.logger.debug("SpeziLLMLocal: Local Mock LLM has finished initializing")
        
        return true
    }
}
