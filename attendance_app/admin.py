from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, BiometricData, Subject, Attendance, AttendanceReport, SystemLog, TrainingDataset, Notification


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'role', 'employee_id', 'student_id', 'is_approved', 'is_active']
    list_filter = ['role', 'is_approved', 'is_active']
    search_fields = ['username', 'email', 'employee_id', 'student_id']
    actions = ['approve_users', 'delete_selected_users']
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Additional Info', {
            'fields': ('role', 'employee_id', 'student_id', 'phone', 'department', 'is_approved', 'profile_image')
        }),
    )
    
    def approve_users(self, request, queryset):
        """Approve selected users"""
        updated = 0
        for user in queryset:
            if not user.is_approved:
                user.is_approved = True
                user.save()
                updated += 1
                
                # Create notification for approved user
                Notification.objects.create(
                    user=user,
                    notification_type='success',
                    title='Account Approved',
                    message='Your account has been approved by an administrator. You can now access all features.',
                    action_url='/dashboard/'
                )
                
                # Log the approval
                SystemLog.objects.create(
                    user=request.user,
                    log_type='info',
                    action='User Approved',
                    description=f'Admin approved user account: {user.username}',
                    ip_address=request.META.get('REMOTE_ADDR')
                )
        
        self.message_user(request, f'{updated} user(s) approved successfully.')
    approve_users.short_description = "Approve selected users"
    
    def delete_selected_users(self, request, queryset):
        """Delete selected users with safety check"""
        deleted = 0
        skipped = 0
        
        for user in queryset:
            # Prevent deleting yourself
            if user.id == request.user.id:
                skipped += 1
                continue
            
            username = user.username
            user.delete()
            deleted += 1
            
            # Log the deletion
            SystemLog.objects.create(
                user=request.user,
                log_type='warning',
                action='User Deleted',
                description=f'Admin deleted user account: {username}',
                ip_address=request.META.get('REMOTE_ADDR')
            )
        
        if deleted > 0:
            self.message_user(request, f'{deleted} user(s) deleted successfully.')
        if skipped > 0:
            self.message_user(request, f'{skipped} user(s) skipped (cannot delete yourself).', level='warning')
    delete_selected_users.short_description = "Delete selected users"



@admin.register(BiometricData)
class BiometricDataAdmin(admin.ModelAdmin):
    list_display = ['user', 'biometric_type', 'is_active', 'created_at']
    list_filter = ['biometric_type', 'is_active']
    search_fields = ['user__username']


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ['code', 'name', 'teacher', 'created_at']
    search_fields = ['code', 'name']
    list_filter = ['teacher']


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['user', 'subject', 'date', 'time', 'status', 'recognition_type', 'confidence_score']
    list_filter = ['status', 'recognition_type', 'date']
    search_fields = ['user__username', 'subject__name']
    date_hierarchy = 'date'


@admin.register(AttendanceReport)
class AttendanceReportAdmin(admin.ModelAdmin):
    list_display = ['title', 'report_type', 'start_date', 'end_date', 'generated_by', 'created_at']
    list_filter = ['report_type', 'created_at']
    search_fields = ['title']


@admin.register(SystemLog)
class SystemLogAdmin(admin.ModelAdmin):
    list_display = ['log_type', 'action', 'user', 'ip_address', 'created_at']
    list_filter = ['log_type', 'created_at']
    search_fields = ['action', 'description']
    readonly_fields = ['created_at']


@admin.register(TrainingDataset)
class TrainingDatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'dataset_type', 'total_samples', 'is_trained', 'accuracy', 'created_at']
    list_filter = ['dataset_type', 'is_trained']
    search_fields = ['name']


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['user', 'notification_type', 'title', 'is_read', 'created_at']
    list_filter = ['notification_type', 'is_read', 'created_at']
    search_fields = ['title', 'message', 'user__username']
    readonly_fields = ['created_at']

