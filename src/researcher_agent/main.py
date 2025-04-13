from crewai.flow.flow import Flow, listen, start, router  # type: ignore
from litellm import completion
from researcher_agent.crews.research_crew.research_crew import Research_Crew  # type: ignore

class ExampleFlow(Flow):
    
    
    @start()
    def start_researcher(self):
        user_topic = input("Please enter the topic you want to research: ")
        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a comprehensive analysis on the topic: {user_topic}. The analysis should cover historical context, current challenges, and potential implications or solutions related to this topic."
                },
            ],
        )
        return response['choices'][0]['message']['content']
   
     
    @listen(start_researcher)
    def research_work(self,update_topic):
         crew = Research_Crew().crew().kickoff(inputs={"topic":update_topic})
         return crew.raw
     
     
         
def kickoff():
    flow = ExampleFlow()
    result = flow.kickoff()
    print("Research output:",result)



# Run the research process
if __name__ == "__main__":
    kickoff()
