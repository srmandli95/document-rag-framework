import re 
from pathlib import Path
from typing import BinaryIO

from app.config.settings import settings
from app.services.storage_service import StorageService

class LocalStorageService(StorageService):
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or settings.RAW_DOCUMENTS_DIR)

    def save_file(
        self,
        file_obj: BinaryIO,
        user_id: str,
        document_id: str,
        original_file_name: str,
    ) -> dict:
        safe_file_name = self._sanitize_file_name(original_file_name)

        document_dir = self.base_dir / user_id / document_id
        document_dir.mkdir(parents=True, exist_ok=True)

        destination_path = document_dir / safe_file_name

        file_size_bytes = 0

        with destination_path.open("wb") as dest_file:
            while chunk := file_obj.read(1024 * 1024):
                file_size_bytes += len(chunk)
                dest_file.write(chunk)

        return {
            "storage_provider": "local",
            "storage_path": str(destination_path),
            "file_name": safe_file_name,
            "file_size_bytes": file_size_bytes,
        }

    def delete_file(self, storage_path: str) -> None:
        path = Path(storage_path)
        if path.exists() and path.is_file():
            path.unlink()

    def get_file(self, storage_path: str) -> str:
        path = Path(storage_path)
        if path.exists() and path.is_file():
            return str(path)
        raise FileNotFoundError(f"File not found at path: {storage_path}")

    def get_file_path(self, storage_path: str) -> str:
        """Backward-compatible alias for `get_file`.

        Keeps older callers using `get_file_path` working while the
        implementation lives in `get_file`.
        """
        return self.get_file(storage_path)
        
    def _sanitize_file_name(self, file_name:str) -> str:
        
        clean_name = Path(file_name).name
        clean_name = re.sub(r'[^\w\.-]', '_', clean_name)
        return clean_name

    
