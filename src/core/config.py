from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = Field(default="Aerial Multiclass Classifier", alias="APP_NAME")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    
    output_dir: str = Field(default="./output", alias="OUTPUT_DIR")
    dataset_root: str = Field(default="./datasets", alias="DATASET_ROOT")
    model_path: str = Field(default="models/yolov8n_seg/weights/best.pt", alias="MODEL_PATH")
    allowed_mimetypes_str: str = Field(default="image/png,image/jpg,image/jpeg", alias="ALLOWED_MIMETYPES")
    
    cors_origins_str: str = Field(default="http://localhost:3000,http://localhost:8000", alias="CORS_ORIGINS")
    
    log_dir: str = Field(default="./logs", alias="LOG_DIR")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins_str.split(",")]
    
    @property
    def allowed_mimetypes(self) -> List[str]:
        """Parse allowed extensions from comma-separated string."""
        return [mimetype.strip().lower() for mimetype in self.allowed_mimetypes_str.split(",")]

settings = Settings()
