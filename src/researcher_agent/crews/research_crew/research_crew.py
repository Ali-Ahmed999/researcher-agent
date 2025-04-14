import os
from dotenv import load_dotenv
import requests
import yaml
from typing import Dict, Any
from crewai import Agent, Crew, Task, Process  # type: ignore
from crewai.project import CrewBase, agent, task, crew  # type: ignore

# Load environment variables
load_dotenv()

# Define a function to query the Serper API for real-time search results
def get_serper_data(topic: str) -> dict:
    api_key = os.getenv("SERPER_API_KEY")  # Fetch the API key from the ENV file
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    
    search_url = f"https://api.serper.dev/search?q={topic}&api_key={api_key}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")
    
    try:
        data = response.json()
    except ValueError:
        raise ValueError("Failed to decode JSON response from the API.")
    
    return data

# Function to replace {topic} placeholders in the YAML
def replace_placeholders(config: Dict[str, Any], topic: str) -> Dict[str, Any]:
    # Replace placeholders in the role, goal, and backstory
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = value.format(topic=topic)
        elif isinstance(value, dict):
            config[key] = replace_placeholders(value, topic)
    return config

@CrewBase
class Research_Crew:
    agents_config: Dict[str, Any]  # Explicitly declare the type
    tasks_config: Dict[str, Any]  # Explicitly declare the type

    def __init__(self):
        # Define the base directory (relative to this file)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, 'config')

        # Define the paths to the configuration files
        agents_path = os.path.join(config_dir, 'agents.yaml')
        tasks_path = os.path.join(config_dir, 'tasks.yaml')

        # Check if the files exist
        if not os.path.exists(agents_path):
            raise FileNotFoundError(f"Configuration file not found: {agents_path}")
        if not os.path.exists(tasks_path):
            raise FileNotFoundError(f"Configuration file not found: {tasks_path}")

        # Load YAML configurations
        with open(agents_path, 'r') as agents_file:
            self.agents_config = yaml.safe_load(agents_file)  # Load agents configuration
        
        with open(tasks_path, 'r') as tasks_file:
            self.tasks_config = yaml.safe_load(tasks_file)  # Load tasks configuration
        
        # Replace placeholders in agents config
        topic = 'Pakistan Independence'  # Example topic, replace with dynamic input
        self.agents_config = replace_placeholders(self.agents_config, topic)
        
        # Initialize agents and tasks
        self.agents = [self.research_officer(), self.research_associate()]
        self.tasks = [self.research_task, self.reporting_task]

    @agent
    def research_officer(self) -> Agent:
        return Agent(
            config=self.agents_config['research_officer'],
            verbose=True,
            llm="gemini/gemini-1.5-flash"
        )

    @agent
    def research_associate(self) -> Agent:
        return Agent(
            config=self.agents_config['research_associate'],
            verbose=True,
            llm="gemini/gemini-1.5-flash"
        )

    @task
    def research_task(self, topic: str) -> Task:
        task_config: Dict[str, str] = self.tasks_config['research_task']
        task_config['description'] = f"Conduct thorough research about {topic}"
        task_config['expected_output'] = f"A list with 10 bullet points about {topic}"
        
        serper_data = get_serper_data(topic)
        relevant_info = self.extract_relevant_info(serper_data)
        task_config['description'] += "\nRelevant search results:\n" + "\n".join(relevant_info)
        
        return Task(config=task_config)  # type: ignore

    @task
    def reporting_task(self, topic: str) -> Task:
        task_config: Dict[str, str] = self.tasks_config['reporting_task']
        task_config['description'] = f"Review the context you got for {topic} and expand each topic into a full section for a report."
        task_config['expected_output'] = f"A fully fleshed-out report with sections about {topic}"
        
        serper_data = get_serper_data(topic)
        relevant_info = self.extract_relevant_info(serper_data)
        task_config['description'] += "\nRelevant search results:\n" + "\n".join(relevant_info)
        
        return Task(config=task_config)  # type: ignore

    @crew
    def crew(self, topic: str) -> Crew:
        # Ensure topic is passed to tasks dynamically
        return Crew(
            agents=self.agents,
            tasks=[
                self.research_task(topic),  # Pass topic to research_task
                self.reporting_task(topic)   # Pass topic to reporting_task
            ],
            process=Process.sequential,
            verbose=True
        )

    def extract_relevant_info(self, serper_data: dict) -> list:
        relevant_info = []
        for result in serper_data.get("organic_results", []):
            title = result.get("title")
            link = result.get("link")
            snippet = result.get("snippet")
            relevant_info.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")
        return relevant_info
