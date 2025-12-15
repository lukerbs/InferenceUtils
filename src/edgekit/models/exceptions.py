"""
Custom exceptions for edgekit models module.
"""


class PreflightValidationError(ValueError):
    """Raised when preflight validation cannot proceed due to missing required model metadata."""
    
    def __init__(self, message: str, model_id: str = None):
        """
        Initialize PreflightValidationError.
        
        Args:
            message: Error message describing what metadata is missing
            model_id: Optional model identifier for better error messages
        """
        self.model_id = model_id
        full_message = f"Preflight validation failed: {message}"
        if model_id:
            full_message += f" (model: {model_id})"
        super().__init__(full_message)
