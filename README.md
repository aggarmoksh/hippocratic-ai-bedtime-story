# Hippocratic AI Coding Assignment - Bedtime Storyteller

Welcome to my submission for the Hippocratic AI coding assignment! 

## Project Overview
I transformed the basic skeleton script into a fully functional **Multi-Agent Workflow** using an Actor-Critic architecture. The system is designed to generate high-quality, age-appropriate (5-10 years) bedtime stories through a collaborative AI pipeline.

### System Architecture
Please see the `DIAGRAM.md` file for a visual representation of the workflow. The pipeline consists of:
1. **Intent Classifier:** Categorizes the user's prompt (e.g., Adventure, Educational) to set the tone.
2. **Storyteller Agent:** Generates an initial draft based on the prompt.
3. **LLM Judge Loop:** A strict editor persona reviews the draft specifically for age-appropriateness, pacing, and story arc. It generates a score and feedback.
4. **Human-in-the-Loop (HITL):** Allows the user to provide custom feedback to tweak the story.
5. **Bonus - Parent's Guide:** Generates read-aloud tips, a moral, and discussion questions for parents.

## How to Run Locally

1. Clone this repository to your local machine.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/Scripts/activate