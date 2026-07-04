from pydantic import Field

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=WORKSPACE_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Iris Autonomous Agent API"
    llm_provider: str = Field(alias="IRIS_LLM_PROVIDER", default="mock")
    llm_model: str = Field(alias="IRIS_LLM_MODEL", default="mock-planner")
    max_plan_tasks: int = Field(default=6, alias="IRIS_MAX_PLAN_TASKS")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")


settings = Settings()