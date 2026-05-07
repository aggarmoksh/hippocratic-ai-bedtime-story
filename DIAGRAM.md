# System Architecture Diagram

This diagram outlines the flow of the "Maker-Checker" agentic system used to generate the story.

### Mermaid Diagram
*(If viewing on GitHub, this will render as a visual flowchart)*

```mermaid
graph TD
    A[User Input] --> B[Storyteller Agent]
    B -->|Generates First Draft| C[Judge Agent]
    C -->|Evaluates Draft vs Age 5-10 Criteria| D{Judge Output}
    D -->|Generates Feedback| E[Revisor Agent]
    B -.->|Passes Draft| E
    E -->|Applies Feedback & Rewrites| F[Final Story]
    F --> G[Output to User]