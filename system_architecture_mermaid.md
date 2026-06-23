# System Architecture Diagram

Here is the professional System Architecture Diagram outlining the entire technology stack and data flow for your AI Face Recognition Attendance System.

You can paste this Mermaid code directly into GitHub, Notion, or any Markdown-compatible documentation tool.

```mermaid
flowchart TD
    classDef layer fill:#f8fafc,stroke:#334155,stroke-width:2px,stroke-dasharray: 5 5,color:#1e293b
    classDef client fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef backend fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef ai fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef data fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#4c1d95

    subgraph PresentationLayer ["1. Presentation Layer"]
        UI["Web Browser UI"] ::: client
        Mobile["Mobile Device Portal"] ::: client
        Cam["Hardware Camera"] ::: client
    end

    subgraph ApplicationLayer ["2. Application Layer"]
        Auth["Auth and Roles Module"] ::: backend
        Routing["URL Routing and Views"] ::: backend
        AttendanceMgr["Attendance Manager"] ::: backend
        Reporting["Reporting Engine"] ::: backend
        
        Routing --> Auth
        Routing --> AttendanceMgr
        Routing --> Reporting
    end

    subgraph AILayer ["3. AI and Computer Vision Layer"]
        PreProcess["Image Preprocessing"] ::: ai
        Detect["Face Detection"] ::: ai
        Extract["Feature Extraction"] ::: ai
        Match["Biometric Matching"] ::: ai
        
        PreProcess --> Detect
        Detect --> Extract
        Extract --> Match
    end

    subgraph DataLayer ["4. Data and Storage Layer"]
        DB[("Relational DB")] ::: data
        Media[("Media Storage")] ::: data
        ModelStore[("Model Storage")] ::: data
    end

    %% Connections Between Layers
    UI <-->|HTTP Request| Routing
    Mobile <-->|HTTP Request| Routing
    Cam -->|Video Stream| Routing

    AttendanceMgr <-->|Frame data| PreProcess
    AttendanceMgr <-->|Confidence Score| Match

    Auth <--> DB
    AttendanceMgr <--> DB
    Reporting <--> DB
    
    PreProcess -.-> Media
    Match <--> ModelStore
    Match <--> DB

    class PresentationLayer layer
    class ApplicationLayer layer
    class AILayer layer
    class DataLayer layer
```
