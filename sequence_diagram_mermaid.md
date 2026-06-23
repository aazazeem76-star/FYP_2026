# System Sequence Diagram

Here is the professional Sequence Diagram detailing the exact step-by-step workflow of a student marking their attendance using your AI Face Recognition system.

You can paste this Mermaid code directly into GitHub, Notion, or any Markdown-compatible documentation tool.

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant UI as Web/Camera Interface
    participant Django as Django Application Server
    participant AI as Face Recognition Module
    participant DB as Database (SQLite3)

    note over Student, DB: Phase 1: Frame Capture & Submission
    Student->>UI: Position face in front of camera
    UI->>UI: Capture Video Frame
    UI->>Django: POST Frame Data (Base64/Multipart)

    note over Django, AI: Phase 2: AI Face Processing
    Django->>AI: extract_face_encoding(frame)
    activate AI
    AI->>AI: apply_clahe() for lighting normalization
    AI->>AI: detect_faces() via Haar Cascades
    AI->>AI: preprocess_face() resize & normalize
    AI-->>Django: Return 128D Face Encoding array
    deactivate AI

    note over Django, DB: Phase 3: Biometric Verification
    Django->>DB: Query BiometricData for Enrolled Students
    activate DB
    DB-->>Django: Return Registered Encodings
    deactivate DB
    Django->>Django: Calculate similarity / Confidence Score
    
    alt Match Confidence > Threshold
        note over Django, DB: Phase 4: Record Attendance
        Django->>DB: Check LiveAttendanceSession validity
        Django->>DB: Create Attendance (status='present', locked=True)
        activate DB
        DB-->>Django: Save Success
        deactivate DB
        Django-->>UI: Return JSON {success: true, student_name: "..."}
        UI-->>Student: Display "Attendance Marked!" & Redirect
    else Match Failed / No Face Detected
        Django-->>UI: Return JSON {success: false, error: "No Match"}
        UI-->>Student: Display "Authentication Failed, Try Again"
    end
```
