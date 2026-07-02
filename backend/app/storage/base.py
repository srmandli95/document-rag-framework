from abc import ABC, abstractmethod
from typing import BinaryIO

class StorageService(ABC):
    """Abstract interface for document storage backends."""

    @abstractmethod
    def save_file(
        self,
        file_obj: BinaryIO,
        user_id: str,
        document_id: str,
        original_file_name: str,
    ) -> dict:
        """Persist a file-like object and return storage metadata."""
        pass

    @abstractmethod
    def delete_file(
        self,
        storage_path: str
    ) -> None:
        """Remove a stored file identified by its storage path."""
        pass

    @abstractmethod
    def get_file(
        self,
        storage_path: str
    ) -> str:
        """Return a readable path or locator for a stored file."""
        pass

        
