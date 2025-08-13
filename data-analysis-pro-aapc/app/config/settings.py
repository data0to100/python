import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    environment: str = "dev"
    log_level: str = "INFO"
    default_model: str = "power_tool"
    model_artifacts_dir: str = "artifacts"

    database_url: Optional[str] = None
    hubspot_api_key: Optional[str] = None

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            environment=os.getenv("ENV", "dev"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            default_model=os.getenv("DEFAULT_MODEL", "power_tool"),
            model_artifacts_dir=os.getenv("MODEL_ARTIFACTS_DIR", "artifacts"),
            database_url=os.getenv("DATABASE_URL"),
            hubspot_api_key=os.getenv("HUBSPOT_API_KEY"),
        )


settings = Settings.load()