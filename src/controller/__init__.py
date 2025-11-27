"""Translation pipeline controller and orchestration.

This module handles the orchestration of the multi-agent translation
pipeline (EN→FR→HE→EN), managing agent invocations and result storage.

Main components:
    - TranslationPipelineController: Main orchestration class
    - AgentInvoker: Handles individual agent invocations
    - ResultManager: Manages result storage and persistence
"""
