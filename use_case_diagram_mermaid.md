# System Use Case Diagram

Here is the professional Use Case Diagram outlining the exact capabilities and interactions of each Actor (Student, Teacher, Admin) with the AI Face Recognition Attendance System.

You can paste this Mermaid code directly into GitHub, Notion, or any Markdown-compatible documentation tool.

```mermaid
flowchart LR
    %% Styling
    classDef actor fill:#f8fafc,stroke:#334155,stroke-width:2px,color:#0f172a,font-weight:bold
    classDef usecase fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e,border-radius:50px
    classDef systemBoundary fill:#ffffff,stroke:#94a3b8,stroke-width:2px,stroke-dasharray: 5 5,color:#1e293b,font-weight:bold

    %% Actors
    Student(["👤 Student"]) ::: actor
    Teacher(["🧑‍🏫 Teacher"]) ::: actor
    Admin(["🛠️ System Admin"]) ::: actor

    %% System Boundary
    subgraph System ["AI Face Recognition Attendance System"]
        direction TB
        
        %% Common
        Login([Secure Login / Auth]) ::: usecase

        %% Student Use Cases
        UC1([Mark Attendance via Face]) ::: usecase
        UC2([View Own Attendance History]) ::: usecase
        UC3([Provide Biometric Samples]) ::: usecase
        
        %% Teacher Use Cases
        UC4([Start Live Attendance Session]) ::: usecase
        UC5([Monitor Class Attendance]) ::: usecase
        UC6([Manual Attendance Override]) ::: usecase
        UC7([Generate Subject Reports]) ::: usecase
        
        %% Admin Use Cases
        UC8([Manage Users & Approvals]) ::: usecase
        UC9([Train AI Recognition Models]) ::: usecase
        UC10([Manage Departments/Subjects]) ::: usecase
        UC11([Configure System Cameras]) ::: usecase
        UC12([View System Activity Logs]) ::: usecase
    end

    %% Connections - Common
    Student --> Login
    Teacher --> Login
    Admin --> Login

    %% Connections - Student
    Student --> UC1
    Student --> UC2
    Student --> UC3

    %% Connections - Teacher
    Teacher --> UC4
    Teacher --> UC5
    Teacher --> UC6
    Teacher --> UC7

    %% Connections - Admin
    Admin --> UC8
    Admin --> UC9
    Admin --> UC10
    Admin --> UC11
    Admin --> UC12

    %% Includes / Extends (Dashed arrows)
    UC1 -.->|"<<include>>"| Login
    UC4 -.->|"<<include>>"| Login
    UC8 -.->|"<<include>>"| Login
    
    class System systemBoundary
```
