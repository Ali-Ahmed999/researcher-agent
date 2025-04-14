from crewai.flow.flow import Flow, listen, start  # type: ignore
from litellm import completion
from researcher_agent.crews.research_crew.research_crew import Research_Crew  # type: ignore

class ExampleFlow(Flow):

    @start()
    def start_researcher(self):
        # Get user input for the topic
        user_topic = input("Please enter the topic you want to research: ")
        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[{
                "role": "user",
                "content": f"Generate a comprehensive analysis on the topic: {user_topic}. The analysis should cover historical context, current challenges, and potential implications or solutions related to this topic."
            }],
        )
        return user_topic  # Return the topic to the next step
    
    @listen(start_researcher)
    def research_work(self, update_topic):
        print(f"Received topic: {update_topic}")  # Debugging line to check topic
        crew = Research_Crew().crew()  # Pass topic here
        result = crew.kickoff(inputs={"topic" : update_topic})  # Execute the tasks
        return result.raw

def kickoff():
    flow = ExampleFlow()
    result = flow.kickoff()  # Start the flow execution
    print("Research output:", result)

# Run the research process
if __name__ == "__main__":
    kickoff()
