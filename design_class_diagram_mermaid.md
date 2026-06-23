# Design Class Diagram

Here is the complete and professional Design Class Diagram for your AI Face Recognition Attendance System. 

It includes the classes, attributes, methods, and relationships mapping exactly to your Django backend (`models.py`) and your AI Recognition script (`face_recognition.py`).

You can paste this Mermaid code directly into GitHub, Notion, or any Markdown-compatible documentation tool.

```mermaid
classDiagram
    %% Core Logic Classes
    class Department {
        +int id
        +string name
        +string code
        +string description
        +__str__() string
        +section_count() int
        +student_count() int
        +subject_count() int
    }

    class Section {
        +int id
        +string name
        +int department_id
        +__str__() string
        +student_count() int
        +subject_count() int
    }

    class User {
        +int id
        +string username
        +string role
        +string student_id
        +string employee_id
        +string phone
        +bool is_approved
        +int dept_id
        +int section_id
        +__str__() string
    }

    class Subject {
        +int id
        +string code
        +string name
        +string description
        +int teacher_id
        +__str__() string
        +student_count() int
    }

    class Attendance {
        +int id
        +date date
        +time time
        +string status
        +string recognition_type
        +float confidence_score
        +bool is_verified
        +bool is_locked
        +int user_id
        +int subject_id
        +__str__() string
    }

    %% AI / Hardware Components
    class FaceRecognition {
        +string model_path
        +object model
        +dict label_encoder
        +tuple img_size
        +object face_cascade
        +load_cascade() void
        +apply_clahe(image) image
        +normalize_brightness(image) image
        +detect_faces(image) list
        +preprocess_face(face_img) array
        +extract_face_encoding(image) array
    }

    class TrainingDataset {
        +int id
        +string name
        +string dataset_type
        +int total_samples
        +bool is_trained
        +string model_path
        +float accuracy
        +update_sample_count() void
        +__str__() string
    }

    class Camera {
        +int id
        +string name
        +string camera_url
        +string camera_type
        +bool is_active
        +int resolution_width
        +int resolution_height
        +int fps
        +get_resolution_display() string
        +__str__() string
    }

    class LiveAttendanceSession {
        +int id
        +datetime started_at
        +bool is_active
        +int subject_id
        +int started_by_id
        +__str__() string
    }

    %% Relationships & Multiplicities
    Department "1" *-- "*" Section : contains
    Department "1" o-- "*" User : has_members
    Section "1" o-- "*" User : categorizes
    Department "1" o-- "*" Subject : offers
    User "1" o-- "*" Subject : teaches (teacher)
    Subject "*" -- "*" User : enrolled_in (students)
    
    User "1" *-- "*" Attendance : logs
    Subject "1" *-- "*" Attendance : tracks
    
    TrainingDataset "1" ..> "1" FaceRecognition : generates_model_for
    Subject "1" -- "0..1" LiveAttendanceSession : hosts
    Camera "1" ..> "*" LiveAttendanceSession : provides_feed_for
```
