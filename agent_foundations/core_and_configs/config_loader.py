# config_loader.py
import yaml
import os
from pydantic import BaseModel, Field, ValidationError, SecretStr
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# --- Pydantic Models ---
# These classes define the expected structure and types of our configuration.

class LLMSettings(BaseModel):
    """LLM parameters that can be overridden per agent."""
    temperature: Optional[float] = None
    model: Optional[str] = None
    # Add other LLM params like max_tokens if needed for other models

class LLMProviderConfig(BaseModel):
    """Configuration for the LLM service provider."""
    type: str = "gemini"        # Default to gemini
    model: Optional[str] = None # Default model for the provider
    api_key: SecretStr          # Use SecretStr for sensitive data like API keys

class AgentConfig(BaseModel):
    """Configuration specific to a single CrewAI agent."""
    role: str
    goal: str
    backstory: str
    tools: List[str] = []       # List of tool names
    allow_delegation: bool = False
    verbose: bool = True
    llm_config_override: Optional[LLMSettings] = None # Optional agent-specific LLM settings

class AppConfig(BaseModel):
    """The root configuration model."""
    default_llm_settings: LLMSettings = Field(default_factory=LLMSettings)
    llm_provider: LLMProviderConfig
    agents: Dict[str, AgentConfig] # Dictionary mapping agent names to their configs

# --- Loading Function ---

def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> AppConfig:
    """Loads configuration from YAML and .env files and validates it."""
    print(f"Loading environment variables from: {env_path}")
    load_dotenv(dotenv_path=env_path) # Load .env file into environment variables

    # Get the API key from environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables (.env file).")

    print(f"Loading YAML configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file: {e}")

    # --- Integrate API Key and Validate ---
    # Manually add the loaded API key to the raw config before validation
    # This is one way to handle secrets alongside file config
    if 'llm_provider' in raw_config:
         raw_config['llm_provider']['api_key'] = google_api_key
    else:
         # Handle case where llm_provider section might be missing? Or assume it exists.
         # For this example, we assume it exists based on our YAML.
         raise ValueError("Missing 'llm_provider' section in config.yaml")


    try:
        # Validate the entire structure using Pydantic
        validated_config = AppConfig(**raw_config)
        print("Configuration loaded and validated successfully.")
        return validated_config
    except ValidationError as e:
        print(f"Configuration validation failed:\n{e}")
        raise e

# --- Example Usage (Optional: Test loading directly) ---
if __name__ == "__main__":
    try:
        config = load_config()
        # Access validated config easily
        print("\n--- Loaded Configuration ---")
        print(f"Default Temp: {config.default_llm_settings.temperature}")
        print(f"LLM Provider Type: {config.llm_provider.type}")
        print(f"LLM Default Model: {config.llm_provider.model}")
        # print(f"LLM API Key: {config.llm_provider.api_key.get_secret_value()}") # Careful printing secrets!
        print(f"Available agent configs: {list(config.agents.keys())}")
        researcher_config = config.agents.get("topic_researcher")
        if researcher_config:
             print(f"Researcher Role: {researcher_config.role}")
             print(f"Researcher Goal: {researcher_config.goal}")
             print(f"Researcher Tools: {researcher_config.tools}")
             if researcher_config.llm_config_override:
                 print(f"Researcher Model Override: {researcher_config.llm_config_override.model}")

    except Exception as e:
        print(f"\nError loading configuration during test: {e}")
