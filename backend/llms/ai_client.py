from typing import final

from groq import AsyncGroq
from app.settings import Settings, settings

@final
class AIClientContainer:
    def __init__(
        self,
        settings: Settings,
    ):
        self.async_groq_client = AsyncGroq(api_key=settings.groq_api_key)


ai_client_container = AIClientContainer(settings)

