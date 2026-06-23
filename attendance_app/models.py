from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone


class Department(models.Model):
    """University Department / Academic Program"""
    name = models.CharField(max_length=200, unique=True)
    code = models.CharField(max_length=20, unique=True, help_text="Short code, e.g. CS, BA, EE")
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'departments'
        ordering = ['name']

    def __str__(self):
        return f"{self.code} — {self.name}"

    def section_count(self):
        return self.sections.count()

    def student_count(self):
        return User.objects.filter(dept=self, role='student').count()

    def subject_count(self):
        return Subject.objects.filter(department=self).count()


class Section(models.Model):
    """Shift-based section within a department"""
    SECTION_CHOICES = [
        ('Morning', 'Morning'),
        ('Shifted-Morning', 'Shifted-Morning'),
        ('Evening', 'Evening'),
        ('Shifted-Evening', 'Shifted-Evening'),
    ]
    department = models.ForeignKey(Department, on_delete=models.CASCADE, related_name='sections')
    name = models.CharField(max_length=30, choices=SECTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'sections'
        unique_together = ['department', 'name']
        ordering = ['department__name', 'name']

    def __str__(self):
        return f"{self.department.code} — {self.name}"

    def student_count(self):
        return User.objects.filter(section=self, role='student').count()

    def subject_count(self):
        return Subject.objects.filter(section=self).count()


class User(AbstractUser):
    """Extended User Model"""
    ROLE_CHOICES = [
        ('student', 'Student'),
        ('teacher', 'Teacher'),
        ('admin', 'Administrator'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='student')
    student_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    employee_id = models.CharField(max_length=50, unique=True, null=True, blank=True)
    # Legacy plain-text department field kept for backward compat; new FK below is preferred
    department = models.CharField(max_length=200, blank=True)
    # New FK-based department & section (used for all new logic)
    dept = models.ForeignKey(
        'Department', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='users', verbose_name='Department'
    )
    section = models.ForeignKey(
        'Section', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='users', verbose_name='Section'
    )
    phone = models.CharField(max_length=15, blank=True)
    phone_number = models.CharField(max_length=15, blank=True)
    address = models.TextField(blank=True)
    is_approved = models.BooleanField(default=False)
    profile_image = models.ImageField(upload_to='profiles/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'users'

    def __str__(self):
        if self.role == 'student' and self.student_id:
            return f"{self.username} ({self.student_id})"
        elif self.role in ['teacher', 'admin'] and self.employee_id:
            return f"{self.username} ({self.employee_id})"
        return self.username


class BiometricData(models.Model):
    """Store biometric data for users"""
    BIOMETRIC_TYPES = [
        ('face', 'Face Recognition'),
        ('retina', 'Retina Recognition'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='biometric_data')
    biometric_type = models.CharField(max_length=20, choices=BIOMETRIC_TYPES)
    image_path = models.ImageField(upload_to='biometric/', help_text="Path to biometric image")
    encoded_data = models.BinaryField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'biometric_data'
        unique_together = ['user', 'biometric_type']
    
    def __str__(self):
        return f"{self.user.username} - {self.biometric_type}"


class Subject(models.Model):
    """Subjects/Courses"""
    name = models.CharField(max_length=200)
    code = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    teacher = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='teaching_subjects')
    # Department & Section this subject belongs to
    department = models.ForeignKey(
        'Department', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='subjects', verbose_name='Department'
    )
    section = models.ForeignKey(
        'Section', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='subjects', verbose_name='Section'
    )
    # Many-to-many relationship with students
    students = models.ManyToManyField(User, related_name='enrolled_subjects', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'subjects'

    def __str__(self):
        dept_section = ''
        if self.department and self.section:
            dept_section = f' [{self.department.code} - {self.section.name}]'
        elif self.department:
            dept_section = f' [{self.department.code}]'
        return f"{self.code} - {self.name}{dept_section}"

    def student_count(self):
        return self.students.count()



class Attendance(models.Model):
    """Attendance Records"""
    STATUS_CHOICES = [
        ('present', 'Present'),
        ('absent', 'Absent'),
        ('late', 'Late'),
    ]
    RECOGNITION_TYPES = [
        ('face', 'Face Recognition'),
        ('retina', 'Retina Recognition'),
        ('manual', 'Manual Entry'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='attendance_records')
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE, related_name='attendance_records', null=True, blank=True)
    date = models.DateField(default=timezone.now)
    time = models.TimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='present')
    recognition_type = models.CharField(max_length=20, choices=RECOGNITION_TYPES, default='manual')
    marked_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='marked_attendance')
    confidence_score = models.FloatField(default=0.0)
    is_verified = models.BooleanField(default=True)
    is_locked = models.BooleanField(
        default=False,
        help_text='Locked when a student self-marks via biometric. Locked records cannot be edited or overridden.',
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'attendance'
        unique_together = ['user', 'subject', 'date']
        ordering = ['-date', '-time']
    
    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.status}"


class AttendanceReport(models.Model):
    """Generated Reports"""
    REPORT_TYPES = [
        ('daily', 'Daily Report'),
        ('weekly', 'Weekly Report'),
        ('monthly', 'Monthly Report'),
        ('custom', 'Custom Report'),
    ]
    
    title = models.CharField(max_length=200)
    report_type = models.CharField(max_length=20, choices=REPORT_TYPES)
    start_date = models.DateField()
    end_date = models.DateField()
    subject = models.ForeignKey(Subject, on_delete=models.SET_NULL, null=True, blank=True)
    generated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    file_path = models.FileField(upload_to='reports/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'reports'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.created_at.date()}"


class SystemLog(models.Model):
    """System Activity Logs"""
    LOG_TYPES = [
        ('info', 'Information'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('security', 'Security'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    log_type = models.CharField(max_length=20, choices=LOG_TYPES, default='info')
    action = models.CharField(max_length=200)
    description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'system_logs'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.action} - {self.created_at}"


class TrainingDataset(models.Model):
    """Training Dataset Management"""
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    dataset_type = models.CharField(max_length=20, choices=[('face', 'Face Recognition'), ('retina', 'Retina Recognition')])
    total_samples = models.IntegerField(default=0)
    is_trained = models.BooleanField(default=False)
    model_path = models.CharField(max_length=500, blank=True)
    accuracy = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'training_datasets'
    
    def __str__(self):
        return f"{self.name} ({self.dataset_type})"
    
    def update_sample_count(self):
        """Update total_samples count from samples linked to this specific dataset"""
        from .models import BiometricSample
        # Count only samples linked to this specific dataset
        self.total_samples = BiometricSample.objects.filter(dataset=self).count()
        self.save()


class BiometricSample(models.Model):
    """Individual biometric samples for training"""
    SAMPLE_TYPE_CHOICES = [
        ('face', 'Face Sample'),
        ('retina', 'Retina Sample'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='biometric_samples')
    sample_type = models.CharField(max_length=20, choices=SAMPLE_TYPE_CHOICES)
    image_path = models.CharField(max_length=500)
    dataset = models.ForeignKey(TrainingDataset, on_delete=models.CASCADE, related_name='samples', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'biometric_samples'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.sample_type} - {self.created_at}"


class Notification(models.Model):
    """User Notifications"""
    NOTIFICATION_TYPES = [
        ('success', 'Success'),
        ('alert', 'Alert'),
        ('error', 'Error'),
        ('info', 'Information'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES, default='info')
    title = models.CharField(max_length=200)
    message = models.TextField()
    action_url = models.CharField(max_length=500, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'notifications'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.title}"


class Camera(models.Model):
    """Camera Configuration for Face/Retina Recognition"""
    CAMERA_TYPE_CHOICES = [
        ('usb', 'USB Camera'),
        ('ip', 'IP Camera'),
        ('rtsp', 'RTSP Stream'),
    ]
    
    name = models.CharField(max_length=200, help_text="Display name for the camera")
    location = models.CharField(max_length=255, help_text="Physical location of the camera")
    camera_url = models.TextField(help_text="Camera URL/Stream or device ID")
    camera_type = models.CharField(max_length=10, choices=CAMERA_TYPE_CHOICES, default='usb')
    is_active = models.BooleanField(default=True, help_text="Enable/disable this camera")
    resolution_width = models.IntegerField(default=640, help_text="Camera resolution width in pixels")
    resolution_height = models.IntegerField(default=480, help_text="Camera resolution height in pixels")
    fps = models.IntegerField(default=30, help_text="Frames per second")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'cameras'
        ordering = ['-is_active', 'name']
    
    def __str__(self):
        return f"{self.name} ({self.location})"
    
    def get_resolution_display(self):
        """Return resolution as 'WIDTHxHEIGHT' string"""
        return f"{self.resolution_width}x{self.resolution_height}"


class LiveAttendanceSession(models.Model):
    """Tracks an active live attendance session started by a teacher for a subject."""
    subject = models.OneToOneField(Subject, on_delete=models.CASCADE, related_name='live_session')
    started_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='live_sessions')
    started_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'live_attendance_sessions'

    def __str__(self):
        status = 'Active' if self.is_active else 'Stopped'
        return f"{self.subject.code} - {self.started_by.username} [{status}]"
