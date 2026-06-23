# Domain Model - Entity Relationship Diagram

Here is the perfect and professional Entity-Relationship diagram for your Face Recognition Attendance System, based directly on your Django `models.py`.

You can copy and paste this code block directly into your `README.md` or any GitHub markdown file.

```mermaid
erDiagram
    %% Core Organization Entities
    Department {
        int id PK
        string name
        string code
        string description
    }
    
    Section {
        int id PK
        string name "Morning/Shifted-Morning/Evening"
        int department_id FK
    }

    %% User & Authentication
    User {
        int id PK
        string username
        string role "student/teacher/admin"
        string student_id
        string employee_id
        string phone
        boolean is_approved
        int dept_id FK
        int section_id FK
    }

    %% Academic Entities
    Subject {
        int id PK
        string code
        string name
        int teacher_id FK
        int department_id FK
        int section_id FK
    }

    %% Core Business Logic
    Attendance {
        int id PK
        date date
        time time
        string status "present/absent/late"
        string recognition_type "face/retina/manual"
        float confidence_score
        boolean is_verified
        boolean is_locked
        int user_id FK
        int subject_id FK
        int marked_by_id FK
    }

    LiveAttendanceSession {
        int id PK
        datetime started_at
        boolean is_active
        int subject_id FK
        int started_by_id FK
    }

    %% Biometric & AI Model Data
    BiometricData {
        int id PK
        string biometric_type "face/retina"
        string image_path
        binary encoded_data
        boolean is_active
        int user_id FK
    }

    TrainingDataset {
        int id PK
        string name
        string dataset_type
        int total_samples
        boolean is_trained
        string model_path
        float accuracy
    }

    BiometricSample {
        int id PK
        string sample_type
        string image_path
        int user_id FK
        int dataset_id FK
    }

    %% System Data
    Camera {
        int id PK
        string name
        string camera_url
        string camera_type "usb/ip/rtsp"
        boolean is_active
        int resolution_width
    }

    SystemLog {
        int id PK
        string log_type
        string action
        string description
        string ip_address
        int user_id FK
    }

    %% Relationships
    Department ||--o{ Section : "has"
    Department ||--o{ User : "contains"
    Department ||--o{ Subject : "offers"
    
    Section ||--o{ User : "groups"
    Section ||--o{ Subject : "scheduled_in"
    
    User ||--o{ BiometricData : "owns"
    User ||--o{ BiometricSample : "provides"
    User ||--o{ Attendance : "marked_in"
    User ||--o{ Subject : "teaches (if teacher)"
    User ||--o{ LiveAttendanceSession : "starts"
    User ||--o{ SystemLog : "generates"
    
    Subject ||--o{ Attendance : "records"
    Subject ||--o| LiveAttendanceSession : "hosts"
    Subject }o--o{ User : "students_enrolled"

    TrainingDataset ||--o{ BiometricSample : "contains"
```
