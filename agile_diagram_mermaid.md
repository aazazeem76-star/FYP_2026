# Agile Development Diagrams

Here are two professional Mermaid diagrams representing the Agile Development lifecycle for your AI-Based Face Recognition Attendance System. You can copy and paste these into your GitHub `README.md` or any other Markdown-compatible documentation.

### 1. Cyclical Flowchart
This layout represents the continuous, looping nature of the Agile cycle.

```mermaid
flowchart LR
    %% Styles
    classDef phase1 fill:#e0f2fe,stroke:#2563eb,stroke-width:2px,color:#1e3a8a,font-weight:bold
    classDef phase2 fill:#ccfbf1,stroke:#0d9488,stroke-width:2px,color:#115e59,font-weight:bold
    classDef phase3 fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d,font-weight:bold
    classDef phase4 fill:#fef9c3,stroke:#eab308,stroke-width:2px,color:#713f12,font-weight:bold
    classDef phase5 fill:#ffedd5,stroke:#ea580c,stroke-width:2px,color:#7c2d12,font-weight:bold
    classDef phase6 fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87,font-weight:bold
    classDef center fill:#ffffff,stroke:#4f46e5,stroke-width:3px,color:#312e81,font-size:16px,font-weight:bold

    Center(("Agile Development<br/>for<br/>AI-Based Face Recognition<br/>Attendance System")):::center

    P1["1. Requirements Analysis<br/>• Identify problems<br/>• Collect requirements<br/>• Define objectives<br/>• Feasibility study"]:::phase1
    P2["2. Planning & Design<br/>• Use Case Design<br/>• System Architecture<br/>• Database Design<br/>• UI/UX Planning"]:::phase2
    P3["3. Development (Implementation)<br/>• User Authentication<br/>• Role Management<br/>• Face Recognition<br/>• Attendance Management<br/>• Reports & Logs"]:::phase3
    P4["4. Testing<br/>• Unit Testing<br/>• Integration Testing<br/>• System Testing<br/>• Performance Testing<br/>• User Acceptance"]:::phase4
    P5["5. Deployment<br/>• Deploy System<br/>• Data Migration<br/>• User Training<br/>• Go-Live"]:::phase5
    P6["6. Review & Improvement<br/>• Monitor System<br/>• Collect Feedback<br/>• Fix Issues<br/>• Enhance Features"]:::phase6

    %% Circular Connections
    P1 ==> P2 ==> P3 ==> P4 ==> P5 ==> P6 ==> P1
    
    %% Connect Center
    Center -.- P1
    Center -.- P2
    Center -.- P3
    Center -.- P4
    Center -.- P5
    Center -.- P6
```

### 2. Radial Mindmap
This layout provides a branching, centralized view of the phases, perfect for highlighting details.

```mermaid
mindmap
  root((Agile Development<br/>for AI-Based Face<br/>Recognition<br/>Attendance System))
    (1. Requirements Analysis)
      Identify problems
      Collect requirements
      Define objectives
      Feasibility study
    (2. Planning & Design)
      Use Case Design
      System Architecture
      Database Design
      UI/UX Planning
    (3. Development)
      User Authentication
      Role Management
      Face Recognition Module
      Attendance Management
      Reports & Audit Logs
    (4. Testing)
      Unit Testing
      Integration Testing
      System Testing
      Performance Testing
      User Acceptance Testing
    (5. Deployment)
      Deploy System
      Data Migration
      User Training
      Go-Live
    (6. Review & Improvement)
      Monitor System
      Collect Feedback
      Fix Issues
      Enhance Features
```
