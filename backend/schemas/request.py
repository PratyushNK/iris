from pydantic import BaseModel, Field, field_validator


class UserRequest(BaseModel):
    request: str = Field(
        min_length=3,
        max_length=8000,
        description="Natural language request for the autonomous agent",
    )

    @field_validator("request")
    @classmethod
    def normalize_request(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("request cannot be empty")
        return normalized