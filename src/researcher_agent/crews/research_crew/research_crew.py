import os
from dotenv import load_dotenv
import requests
import yaml
from typing import Dict, Any
from crewai import Agent, Crew, Task, Process  # type: ignore
from crewai.project import CrewBase, agent, task, crew  # type: ignore
from crewai.tools import BaseTool # type: ignore
from crewai_tools import SerperDevTool # type: ignore

google_search_tool = SerperDevTool()

load_dotenv()


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
    def research_task(self) -> Task:
        task_config: Dict[str, str] = self.tasks_config['research_task']
        task_config['description'] = "Conduct thorough research about {topic}"
        task_config['expected_output'] = "A list with 10 bullet points about {topic}"
        
        return Task(config=task_config, tools=[google_search_tool])  # type: ignore

    @task
    def reporting_task(self) -> Task:
        task_config: Dict[str, str] = self.tasks_config['reporting_task']
        task_config['description'] = "Review the context you got for {topic} and expand each topic into a full section for a report."
        task_config['expected_output'] = "A fully fleshed-out report with sections about {topic}"
        
        
        return Task(config=task_config, tools=[google_search_tool])  # type: ignore

    @crew
    def crew(self) -> Crew:
        # Ensure topic is passed to tasks dynamically
        return Crew(
            agents=self.agents,
            tasks=[
                self.research_task(),  # Pass topic to research_task
                self.reporting_task()   # Pass topic to reporting_task
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
