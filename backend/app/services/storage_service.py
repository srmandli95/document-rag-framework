from abc import ABC, abstractmethod
from tying import BinaryIO

class StorageService(ABC):
    """
    Abstact storage interface.

    """

    @abstractmethod
    def save_file(
        self,
        file_obj: BinaryIO,
        user_id: str,
        document_id: str,
        original_file_name: str,
    ) -> dict:
        pass

    @abstractmethod
    def delete_file(
        self,
        storage_path: str
    ) -> None:
        pass

    @abstractmethod
    def get_file(
        self,
        storage_path: str
    ) -> str:
        pass

        