from crewai import Agent, Crew, Task, Process  # type: ignore
from crewai.project import CrewBase, agent, task, crew # type: ignore

@CrewBase
class Research_Crew:
    
    agents_config = 'config/agents.yaml' 
    tasks_config = 'config/tasks.yaml' 
    
    
    @agent
    def research_officer(self) -> Agent:
        return Agent(
            config=self.agents_config['research_officer'], # type: ignore
            verbose=False,
            llm="gemini/gemini-1.5-flash"
        )
        
        
        
    @agent
    def research_associate(self) -> Agent:
        return Agent(
            config=self.agents_config['research_associate'], # type: ignore
            verbose=False,
            llm="gemini/gemini-1.5-flash"
        )
        
        
    
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']# type: ignore
        )
        
        
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'] # type: ignore
        )
        
        
        
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # type: ignore # Automatically collected by the @agent decorator
            tasks=self.tasks,  # type: ignore   # Automatically collected by the @task decorator. 
            process=Process.sequential,
            verbose=False,
        )