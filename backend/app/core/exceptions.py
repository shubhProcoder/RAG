"""
Custom exception hierarchy for the PDF RAG application.

All application exceptions inherit from AppError, enabling
consistent error handling at the API boundary.
"""


class AppError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class DocumentNotFoundError(AppError):
    """Raised when a requested document does not exist."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(
            message=f"Document not found: {document_id}",
            detail=f"No document exists with ID '{document_id}'",
        )


class ExtractionError(AppError):
    """Raised when PDF text extraction fails."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        super().__init__(
            message=f"Failed to extract text from '{filename}'",
            detail=reason,
        )


class StorageError(AppError):
    """Raised when a storage operation (vector DB or metadata DB) fails."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        super().__init__(
            message=f"Storage error during '{operation}'",
            detail=reason,
        )


class ValidationError(AppError):
    """Raised when input validation fails (file type, size, etc.)."""

    def __init__(self, field: str, reason: str):
        self.field = field
        super().__init__(
            message=f"Validation failed for '{field}'",
            detail=reason,
        )


class ProcessingError(AppError):
    """Raised when the processing pipeline encounters an unrecoverable error."""

    def __init__(self, document_id: str, stage: str, reason: str):
        self.document_id = document_id
        self.stage = stage
        super().__init__(
            message=f"Processing failed for document '{document_id}' at stage '{stage}'",
            detail=reason,
        )
