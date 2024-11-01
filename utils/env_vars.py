import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel


class EnvironmentVariables(BaseModel):

    ANTHROPIC_API_KEY: str | None
    OPENAI_API_ORG: str | None
    OPENAI_API_KEY: str | None

    @classmethod
    def load_from_env(cls):
        env_file = find_env_file()
        load_dotenv(env_file)

        openai_api_org = os.getenv("OPENAI_API_ORG")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        return cls(
            ANTHROPIC_API_KEY=anthropic_api_key,
            OPENAI_API_ORG=openai_api_org,
            OPENAI_API_KEY=openai_api_key
        )


def find_env_file():
    """
    Find the .env file in root directory.
    """
    current_dir = Path(__file__).parent.resolve()
    env_file = current_dir / ".env"
    if not env_file.is_file():
        raise FileNotFoundError(f"No .env file found in: {current_dir}")
    
    return str(env_file)


ENV = EnvironmentVariables.load_from_env()

print(find_env_file())