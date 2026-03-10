import pandas as pd
import os
from pathlib import Path

class DataLoader:
    """
    DataLoader ngắn gọn để load file CSV và Excel
    """
    
    def __init__(self, filepath: str):
        """
        Khởi tạo với đường dẫn file
        """
        self.filepath = Path(filepath)
        self.extension = self.filepath.suffix.lower()
        
    def load(self) -> pd.DataFrame:
        """
        Load file tự động dựa trên extension
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"File không tồn tại: {self.filepath}")
        
        if self.extension == '.csv':
            return self._load_csv()
        elif self.extension in ['.xlsx', '.xls']:
            return self._load_excel()
        else:
            raise ValueError(f"Định dạng không hỗ trợ: {self.extension}")
    
    def _load_csv(self) -> pd.DataFrame:
        """
        Load CSV với encoding tự động
        """
        encodings = ['utf-8', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                return pd.read_csv(self.filepath, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Không thể đọc file với các encoding: {encodings}")
    
    def _load_excel(self) -> pd.DataFrame:
        """
        Load Excel file
        """
        return pd.read_excel(self.filepath)