from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('', views.index, name='index'),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('verify-otp/', views.verify_otp, name='verify_otp'),
    path('logout/', views.user_logout, name='logout'),

    # Face verification gate (for teacher / employee login)
    path('face-verify/', views.face_verify_login, name='face_verify_login'),
    path('api/verify-face-login/', views.api_verify_face_login, name='api_verify_face_login'),
    
    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # Profile
    path('profile/', views.profile, name='profile'),
    path('upload-biometric/', views.upload_biometric, name='upload_biometric'),
    
    # Attendance
    path('mark-attendance/face/', views.mark_attendance_face, name='mark_attendance_face'),
    path('mark-attendance/retina/', views.mark_attendance_retina, name='mark_attendance_retina'),
    path('attendance/history/', views.attendance_history, name='attendance_history'),
    
    # API endpoints for recognition
    path('api/recognize-face/', views.process_face_recognition, name='process_face_recognition'),
    path('api/recognize-retina/', views.process_retina_recognition, name='process_retina_recognition'),
    
    # Subjects
    path('subjects/', views.subjects_list, name='subjects_list'),
    path('subjects/<int:subject_id>/', views.subject_detail, name='subject_detail'),
    path('subjects/create/', views.create_subject, name='create_subject'),
    path('subjects/delete/<int:subject_id>/', views.delete_subject, name='delete_subject'),
    path('subjects/enroll/<int:subject_id>/', views.enroll_students, name='enroll_students'),
    
    # Reports
    path('reports/', views.reports_list, name='reports_list'),
    path('reports/generate/', views.generate_report, name='generate_report'),
    path('reports/delete/<int:report_id>/', views.delete_report, name='delete_report'),
    
    # Admin
    path('admin-panel/', views.admin_panel, name='admin_panel'),
    path('admin-panel/approve-user/<int:user_id>/', views.approve_user, name='approve_user'),
    path('admin-panel/delete-user/<int:user_id>/', views.delete_user, name='delete_user'),
    
    # Notifications Module
    path('notifications/', views.notifications, name='notifications'),
    path('api/notifications/<int:notification_id>/read/', views.mark_notification_read, name='mark_notification_read'),
    path('api/notifications/<int:notification_id>/delete/', views.delete_notification, name='delete_notification'),
    path('api/notifications/mark-all-read/', views.mark_all_notifications_read, name='mark_all_notifications_read'),
    path('api/notifications/clear-all/', views.clear_all_notifications, name='clear_all_notifications'),
    
    # Dataset & Training Module
    path('dataset-management/', views.dataset_management, name='dataset_management'),
    path('api/dataset/create/', views.create_dataset, name='create_dataset'),
    path('api/dataset/collect-samples/', views.collect_samples, name='collect_samples'),
    path('api/dataset/train/', views.train_model, name='train_model'),
    path('api/dataset/<int:dataset_id>/view/', views.view_dataset_detail, name='view_dataset_detail'),
    path('api/dataset/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),
    path('api/sample/<int:sample_id>/delete/', views.delete_sample, name='delete_sample'),
    path('api/model/<int:model_id>/test/', views.test_model_api, name='test_model'),
    path('api/model/<int:model_id>/retrain/', views.retrain_model_api, name='retrain_model'),
    path('api/model/<int:model_id>/export/', views.export_model_api, name='export_model'),
    
    # Manual Attendance Module
    path('manual-attendance/', views.manual_attendance, name='manual_attendance'),
    path('manual-attendance/mark/', views.manual_mark_attendance, name='manual_mark_attendance'),
    path('manual-attendance/bulk/', views.bulk_mark_attendance, name='bulk_mark_attendance'),
    path('api/attendance/<int:attendance_id>/', views.get_attendance_detail, name='get_attendance_detail'),
    path('api/attendance/<int:attendance_id>/update/', views.update_attendance, name='update_attendance'),
    path('api/attendance/<int:attendance_id>/delete/', views.delete_attendance_record, name='delete_attendance_record'),

    # Camera Management
    path('camera-management/', views.camera_management, name='camera_management'),
    path('api/camera/create/', views.create_camera, name='create_camera'),
    path('api/camera/<int:camera_id>/edit/', views.edit_camera, name='edit_camera'),
    path('api/camera/<int:camera_id>/delete/', views.delete_camera, name='delete_camera'),
    path('api/camera/<int:camera_id>/toggle-status/', views.toggle_camera_status, name='toggle_camera_status'),
    path('api/camera/<int:camera_id>/test/', views.test_camera_connection, name='test_camera_connection'),

    # Teacher Attendance
    path('teacher-attendance/', views.teacher_attendance, name='teacher_attendance'),
    path('teacher-attendance/mark/', views.teacher_mark_attendance, name='teacher_mark_attendance'),
    path('teacher-attendance/bulk/', views.teacher_bulk_mark_attendance, name='teacher_bulk_mark_attendance'),

    # Employee Attendance
    path('employee-attendance/', views.employee_attendance, name='employee_attendance'),
    path('employee-attendance/mark/', views.employee_mark_attendance, name='employee_mark_attendance'),
    path('employee-attendance/bulk/', views.employee_bulk_mark_attendance, name='employee_bulk_mark_attendance'),

    # Live Attendance Session
    path('api/live-attendance/start/<int:subject_id>/', views.start_live_attendance, name='start_live_attendance'),
    path('api/live-attendance/stop/<int:subject_id>/', views.stop_live_attendance, name='stop_live_attendance'),
    path('api/live-attendance/check/', views.check_live_session, name='check_live_session'),

    # Teacher Self Face Attendance
    path('mark-my-attendance/', views.mark_teacher_attendance_face, name='mark_teacher_attendance_face'),
    path('api/teacher-face-attendance/', views.api_teacher_face_attendance, name='api_teacher_face_attendance'),

    # Department Management
    path('departments/', views.department_list, name='department_list'),
    path('departments/create/', views.create_department, name='create_department'),
    path('departments/<int:dept_id>/edit/', views.edit_department, name='edit_department'),
    path('departments/<int:dept_id>/delete/', views.delete_department, name='delete_department'),

    # Section Management
    path('sections/', views.section_list, name='section_list'),
    path('sections/create/', views.create_section, name='create_section'),
    path('sections/<int:section_id>/delete/', views.delete_section, name='delete_section'),

    # AJAX helpers
    path('api/sections-by-department/<int:dept_id>/', views.api_sections_by_department, name='api_sections_by_department'),
    path('api/students-by-section/<int:section_id>/', views.api_students_by_section, name='api_students_by_section'),
    path('api/subjects-by-filter/', views.api_subjects_by_dept_section, name='api_subjects_by_dept_section'),
    path('api/students-by-subject/<int:subject_id>/', views.api_students_by_subject, name='api_students_by_subject'),
]
