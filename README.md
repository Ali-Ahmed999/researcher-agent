# 🧠 Researcher Agent

Researcher Agent is an AI-powered tool designed to assist in generating research topics and conducting research workflows. It leverages advanced AI models and a modular crew-based architecture to streamline research processes.

> **Note**: This agent is specifically trained to generate and discuss topics related to **Pakistan's political issues**.

## ✨ Features

- 🔍 **Topic Generation**: Automatically generates research topics using AI.
- 🛠️ **Research Workflow**: Executes research workflows with a modular crew-based system.
- 🔌 **Extensible Design**: Built with a flexible architecture to accommodate additional functionalities.

## 📂 Project Structure
. ├── .env ├── .gitignore ├── pyproject.toml ├── README.md ├── src/ │ └── researcher_agent/ │ └── main.py


- **`main.py`**: Contains the core logic for the research agent, including the `ExampleFlow` class that handles topic generation and research workflows.

## 🛠️ How It Works

1. **Start Researcher**: The `start_researcher` method generates a research topic using the `litellm` library and the `gemini/gemini-1.5-flash` model. It is trained to focus on **Pakistan's political issues**.
2. **Research Workflow**: The `research_work` method uses the `Research_Crew` class to execute the research workflow based on the generated topic.

### 🚀 Example Code

```python
from researcher_agent.main import kickoff

if __name__ == "__main__":
    kickoff()

1. Clone the repository:
git clone https://github.com/Ali-Ahmed999/researcher-agent.git
cd researcher-agent

2.Install dependencies:
pip install -r requirements.txt

3. Set up environment variables in a .env file.
▶️ Usage
Run the project using the following command:
uv run kickoff

🧩 Dependencies
litellm: For AI model integration.
crewai.flow: For flow-based architecture.
Research_Crew: Custom module for research workflows.

🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments
Built with the litellm library for AI-powered topic generation.
Inspired by modular and extensible design principles.
Focused on Pakistan's political issues for research topic generation.
