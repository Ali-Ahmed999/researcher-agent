from crewai.flow.flow import Flow, listen, start, router  # type: ignore
from litellm import completion


class ExampleFlow(Flow):
    
    
    @start()
    def start_researcher(self):
         response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": "generate the name of a random city in pakistan.",
                },
            ],
        )
         return response['choices'][0]['message']['content']

def kickoff():
    flow = ExampleFlow()
    result = flow.kickoff()
    print(result)