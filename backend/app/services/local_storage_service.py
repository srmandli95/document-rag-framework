import re 
from pathlib import Path
from typing import BinaryIO

from app.config.settings import settings
from app.services.storage_service import StoregeService

class LocalStorageService(StorageService):
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or setting.RAW_DOCUMENTS_DIR)

    def save_file(
        self,
        file_obj: BinaryIO,
        user_id: str,
        document_id: str,
        original_file_name: str,
    ) -> dict:
        safe_file_name = self._sanitize_file_name(original_file_name)

        document_dir = self.base_dir / user_id / document_id
        document_dir.mkdir(parents=True, exists_ok=True)

        destination_path = document_dir / safe_file_name
