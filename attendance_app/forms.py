from django import forms
from django.contrib.auth.forms import UserCreationForm
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit, Row, Column, Field
from .models import User, BiometricData, Subject, Attendance, Department, Section
from datetime import datetime


class UserRegistrationForm(UserCreationForm):
    """User Registration Form"""
    email = forms.EmailField(required=True)
    dept = forms.ModelChoiceField(
        queryset=Department.objects.all(),
        required=False,
        empty_label="-- Select Department --",
        label="Department",
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_dept'})
    )
    section = forms.ModelChoiceField(
        queryset=Section.objects.none(),   # populated dynamically via AJAX
        required=False,
        empty_label="-- Select Section --",
        label="Section (Students only)",
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_section'})
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'employee_id', 'student_id',
                  'phone', 'role', 'password1', 'password2']
        # Note: dept and section are handled by the custom HTML dropdowns
        # in register.html — they post as name="dept" / name="section" and
        # are saved directly in the verify_otp view via request.POST.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Register', css_class='btn btn-primary'))

        # Auto-generate employee_id and student_id
        if not self.instance.pk:
            today = datetime.now().strftime('%Y%m%d')

            today_employees = User.objects.filter(
                employee_id__startswith=f'EMP-{today}-'
            ).order_by('-employee_id').first()
            if today_employees and today_employees.employee_id:
                last_counter = int(today_employees.employee_id.split('-')[-1])
                counter = last_counter + 1
            else:
                counter = 1
            emp_id = f'EMP-{today}-{counter:03d}'

            today_students = User.objects.filter(
                student_id__startswith=f'STU-{today}-'
            ).order_by('-student_id').first()
            if today_students and today_students.student_id:
                last_counter = int(today_students.student_id.split('-')[-1])
                counter = last_counter + 1
            else:
                counter = 1
            stu_id = f'STU-{today}-{counter:03d}'

            self.fields['employee_id'].initial = emp_id
            self.fields['student_id'].initial = stu_id
            self.fields['employee_id'].widget.attrs['readonly'] = True
            self.fields['employee_id'].widget.attrs['class'] = 'form-control'
            self.fields['student_id'].widget.attrs['readonly'] = True
            self.fields['student_id'].widget.attrs['class'] = 'form-control'

        # If POST data present, populate section queryset based on selected dept
        if 'dept' in self.data:
            try:
                dept_id = int(self.data.get('dept'))
                self.fields['section'].queryset = Section.objects.filter(department_id=dept_id)
            except (ValueError, TypeError):
                pass
        elif self.instance.pk and self.instance.dept:
            self.fields['section'].queryset = Section.objects.filter(department=self.instance.dept)

        # Restrict role choices — Administrator cannot self-register
        self.fields['role'].choices = [
            ('student', 'Student'),
            ('teacher', 'Teacher'),
        ]


class UserProfileForm(forms.ModelForm):
    """User Profile Edit Form"""
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'phone', 'dept', 'section', 'profile_image']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Update Profile', css_class='btn btn-success'))

        # ── Disable dept & section — admin-only fields ──────────────────────
        # Users must not be able to change their own department or section.
        # Mark both widgets disabled so they render greyed-out and unclickable.
        self.fields['dept'].widget.attrs['disabled'] = True
        self.fields['dept'].required = False
        self.fields['section'].widget.attrs['disabled'] = True
        self.fields['section'].required = False

        # Populate section queryset so the current value displays correctly
        if self.instance.pk and self.instance.dept:
            self.fields['section'].queryset = Section.objects.filter(department=self.instance.dept)
        else:
            self.fields['section'].queryset = Section.objects.none()

    def save(self, commit=True):
        """Never overwrite dept/section regardless of POST data."""
        instance = super().save(commit=False)
        if self.instance.pk:
            original = User.objects.get(pk=self.instance.pk)
            instance.dept    = original.dept
            instance.section = original.section
        if commit:
            instance.save()
        return instance


class BiometricDataForm(forms.ModelForm):
    """Biometric Data Upload Form"""
    class Meta:
        model = BiometricData
        fields = ['biometric_type', 'image_path']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Upload Biometric Data', css_class='btn btn-primary'))


class DepartmentForm(forms.ModelForm):
    """Department Creation / Edit Form"""
    class Meta:
        model = Department
        fields = ['name', 'code', 'description']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Computer Science'}),
            'code': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. CS'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }


class SectionForm(forms.ModelForm):
    """Section Creation / Edit Form"""
    class Meta:
        model = Section
        fields = ['department', 'name']
        widgets = {
            'department': forms.Select(attrs={'class': 'form-select', 'id': 'id_section_dept'}),
            'name': forms.Select(attrs={'class': 'form-select'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['department'].queryset = Department.objects.all()
        self.fields['department'].empty_label = '-- Select Department --'


class SubjectForm(forms.ModelForm):
    """Subject Creation/Edit Form"""
    class Meta:
        model = Subject
        fields = ['name', 'code', 'description', 'department', 'section', 'teacher']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control', 'placeholder': ' ', 'id': 'id_name'
            }),
            'code': forms.TextInput(attrs={
                'class': 'form-control', 'placeholder': ' ', 'id': 'id_code'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control', 'placeholder': ' ', 'id': 'id_description', 'rows': 4
            }),
            'department': forms.Select(attrs={
                'class': 'form-select', 'id': 'id_subject_department'
            }),
            'section': forms.Select(attrs={
                'class': 'form-select', 'id': 'id_subject_section'
            }),
            'teacher': forms.Select(attrs={
                'class': 'form-select', 'id': 'id_teacher'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['teacher'].queryset = User.objects.filter(role='teacher', is_approved=True)
        self.fields['teacher'].label_from_instance = lambda obj: (
            f"{obj.first_name} {obj.last_name}" if obj.first_name and obj.last_name else obj.username
        )
        self.fields['teacher'].empty_label = "-- Select a Teacher --"
        self.fields['teacher'].required = False
        self.fields['department'].queryset = Department.objects.all()
        self.fields['department'].empty_label = "-- Select Department --"
        self.fields['department'].required = False
        # Populate section queryset based on selected department
        if 'department' in self.data:
            try:
                dept_id = int(self.data.get('department'))
                self.fields['section'].queryset = Section.objects.filter(department_id=dept_id)
            except (ValueError, TypeError):
                self.fields['section'].queryset = Section.objects.none()
        elif self.instance.pk and self.instance.department:
            self.fields['section'].queryset = Section.objects.filter(department=self.instance.department)
        else:
            self.fields['section'].queryset = Section.objects.none()
        self.fields['section'].empty_label = "-- Select Section --"
        self.fields['section'].required = False


class AttendanceMarkForm(forms.Form):
    """Manual Attendance Marking Form"""
    subject = forms.ModelChoiceField(queryset=Subject.objects.all(), required=True)
    users = forms.ModelMultipleChoiceField(
        queryset=User.objects.filter(role='student', is_approved=True),
        widget=forms.CheckboxSelectMultiple,
        required=True
    )
    status = forms.ChoiceField(choices=Attendance.STATUS_CHOICES, initial='present')
    notes = forms.CharField(widget=forms.Textarea, required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Mark Attendance', css_class='btn btn-primary'))


class ReportGenerationForm(forms.Form):
    """Report Generation Form"""
    report_type = forms.ChoiceField(
        choices=[('daily', 'Daily'), ('weekly', 'Weekly'), ('monthly', 'Monthly'), ('custom', 'Custom')],
        required=True
    )
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), required=True)
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), required=True)
    subject = forms.ModelChoiceField(queryset=Subject.objects.all(), required=False)
    user = forms.ModelChoiceField(queryset=User.objects.filter(role='student'), required=False)
    export_format = forms.ChoiceField(choices=[('pdf', 'PDF'), ('csv', 'CSV')], initial='pdf')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Generate Report', css_class='btn btn-success'))




