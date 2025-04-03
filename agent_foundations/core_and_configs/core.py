# main.py
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool # Example tool
from langchain_google_generativeai import ChatGoogleGenerativeAI # Specific class for Gemini

from config_loader import load_config, AppConfig # Import our loader and config model

# --- 1. Load Configuration ---
try:
    config: AppConfig = load_config()
except Exception as e:
    print(f"Fatal Error: Could not load configuration. Exiting. \n{e}")
    exit()

# --- 2. Set up the LLM (Google Gemini) ---
# Retrieve LLM settings from the validated config
llm_provider_config = config.llm_provider
gemini_api_key = llm_provider_config.api_key.get_secret_value() # Extract secret safely

# Choose the default model from config, fallback if necessary
default_llm_model = llm_provider_config.model or "gemini-pro" # Fallback if not in config

# Instantiate the Gemini LLM object using CrewAI/LangChain integration
# Note: CrewAI often uses LangChain components under the hood for LLM interaction
try:
    default_llm = ChatGoogleGenerativeAI(
        model=default_llm_model,
        google_api_key=gemini_api_key,
        temperature=config.default_llm_settings.temperature or 0.7, # Use default temp
        # convert_system_message_to_human=True # Sometimes needed for Gemini compatibility
        verbose=True # Log LLM interactions
    )
    print(f"Default Gemini LLM initialized with model: {default_llm_model}")
except Exception as e:
    print(f"Fatal Error: Failed to initialize Gemini LLM. Check API key and model name. \n{e}")
    exit()


# --- Helper Function to Create LLM Instance ---
# This helps manage agent-specific LLM overrides
def get_llm_instance(agent_name: str, base_llm: ChatGoogleGenerativeAI) -> ChatGoogleGenerativeAI:
    """Creates or returns an LLM instance, considering agent overrides."""
    agent_cfg = config.agents.get(agent_name)
    if not agent_cfg or not agent_cfg.llm_config_override:
        return base_llm # Return the default LLM if no override

    override = agent_cfg.llm_config_override
    override_model = override.model or base_llm.model # Use agent model or default
    override_temp = override.temperature if override.temperature is not None else base_llm.temperature # Use agent temp or default

    print(f"Creating overridden LLM for agent '{agent_name}' with model: {override_model}, temp: {override_temp}")
    try:
        # Create a new instance with overridden parameters
        return ChatGoogleGenerativeAI(
            model=override_model,
            google_api_key=gemini_api_key, # Reuse the key
            temperature=override_temp,
            verbose=True
            # convert_system_message_to_human=True
        )
    except Exception as e:
        print(f"Warning: Failed to create overridden LLM for {agent_name}. Using default. Error: {e}")
        return base_llm # Fallback to default if override fails


# --- 3. Define Tools (Optional but common) ---
# Tools allow agents to interact with external systems (search, file I/O, etc.)
# For SerperDevTool, you need a SERPER_API_KEY in your .env file
serper_api_key = os.getenv("SERPER_API_KEY")
if serper_api_key:
    search_tool = SerperDevTool()
    print("Search tool (SerperDevTool) initialized.")
    available_tools = {"search_tools": [search_tool]} # Map name in config to tool object(s)
else:
    print("Warning: SERPER_API_KEY not found in .env. Search tool will not be available.")
    available_tools = {}

# --- 4. Create Agents Dynamically from Config ---
print("\n--- Creating Agents ---")
agents_dict = {} # Store created agent objects

for agent_name, agent_cfg in config.agents.items():
    print(f"Initializing Agent: {agent_name}")

    # Get the correct LLM instance (default or overridden)
    agent_llm = get_llm_instance(agent_name, default_llm)

    # Resolve tool names from config to actual tool objects
    agent_tools_list = []
    if agent_cfg.tools:
        for tool_name in agent_cfg.tools:
            tools = available_tools.get(tool_name)
            if tools:
                agent_tools_list.extend(tools) # Add the list of tools associated with the name
            else:
                print(f"Warning: Tool group '{tool_name}' specified for agent '{agent_name}' not found in available_tools.")

    # Create the CrewAI Agent instance using config values
    agents_dict[agent_name] = Agent(
        role=agent_cfg.role,
        goal=agent_cfg.goal,
        backstory=agent_cfg.backstory,
        llm=agent_llm, # Assign the specific LLM instance
        tools=agent_tools_list, # Assign the resolved tools
        allow_delegation=agent_cfg.allow_delegation,
        verbose=agent_cfg.verbose,
        # memory=True # You can enable memory later, requires more setup
    )
    print(f"Agent '{agent_name}' created with Role: {agent_cfg.role}")


# --- 5. Define Task(s) ---
# Tasks are the specific assignments you give to the agents for a particular run.
print("\n--- Defining Tasks ---")

# Example: Research AI Agents and write a blog post about them
topic = "The latest advancements in AI Agents (Summer 2024)"

research_task = Task(
  description=f"Conduct thorough research on '{topic}'. Find key developments, challenges, and future trends. Identify reliable sources.",
  expected_output="A comprehensive research report with bullet points, key findings, and source URLs.",
  agent=agents_dict.get("topic_researcher") # Assign task to the specific agent object
)

writing_task = Task(
  description=f"Using the provided research report on '{topic}', write an engaging and informative blog post (around 500-700 words). Focus on clarity for a general audience.",
  expected_output=f"A well-structured blog post about '{topic}' with an introduction, main body (covering key findings), and conclusion.",
  agent=agents_dict.get("blog_post_writer"),
  context=[research_task] # Specify that this task depends on the output of the research_task
)

# --- 6. Create and Run the Crew ---
print("\n--- Creating and Running the Crew ---")

# Check if agents required for tasks were created
if not research_task.agent or not writing_task.agent:
    print("Fatal Error: One or more agents specified in tasks were not found in the configuration or failed to initialize.")
    exit()

# Form the crew with the agents and tasks
crew = Crew(
  agents=[agents_dict["topic_researcher"], agents_dict["blog_post_writer"]], # List of agent objects
  tasks=[research_task, writing_task],      # List of task objects
  process=Process.sequential, # Tasks run one after another based on context dependencies
  verbose=2 # Set verbosity level for the crew execution (0, 1, or 2)
)

# Kick off the crew's work!
try:
    print(f"\nStarting crew to work on: {topic}...")
    result = crew.kickoff()

    print("\n\n--- Crew Execution Finished ---")
    print("Final Result:")
    print(result) # The output of the last task in the sequence

except Exception as e:
     print(f"\nAn error occurred during crew execution: {e}")

print("\n--- Script Complete ---")
