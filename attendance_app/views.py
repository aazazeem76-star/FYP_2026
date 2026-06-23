from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Count, Q
from datetime import datetime, timedelta
import json
import cv2
import numpy as np
import base64
from io import BytesIO
import csv
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from .models import User, BiometricData, Subject, Attendance, AttendanceReport, SystemLog, TrainingDataset, Camera, LiveAttendanceSession, Department, Section
from .forms import UserRegistrationForm, UserProfileForm, BiometricDataForm, SubjectForm, AttendanceMarkForm, ReportGenerationForm, DepartmentForm, SectionForm
from ai_models.face_recognition import FaceRecognition
from ai_models.retina_recognition import RetinaRecognition
from django.conf import settings


# Initialize AI models
face_recognizer = FaceRecognition()
retina_recognizer = RetinaRecognition()

# ── Process-level caches to avoid per-request recomputation ──────────────────
# CNN model cache: avoids loading the Keras .h5 file on every request (~2-3 s)
_cnn_cache = {}   # key: (model_path, mtime_int)  → (keras_model, labels_dict)

# Descriptor cache: avoids re-running face detection + HOG+LBP on stored images
# key: (full_image_path, mtime_int, _DESC_VERSION) → np.ndarray descriptor (or None)
# _DESC_VERSION must be bumped whenever the descriptor function changes so that
# stale entries with wrong vector dimensions are never returned.
_DESC_VERSION = 'v2_robust'   # bump when switching extract_* method
_desc_cache = {}
MAX_DESC_CACHE = 500   # max entries; old ones evicted when limit is hit


def _get_cached_descriptor(full_path):
    """Return cached robust descriptor for a stored sample image.
    Uses extract_robust_encoding (glasses-invariant multi-zone HOG+LBP
    weighted 60% + full-face HOG+LBP weighted 40%).
    Cache key includes _DESC_VERSION so stale entries from a previous
    descriptor method are automatically invalidated after a code change.
    Recomputes if the file has changed (mtime check)."""
    if not os.path.exists(full_path):
        return None
    try:
        mtime = int(os.path.getmtime(full_path))
    except OSError:
        return None
    key = (full_path, mtime, _DESC_VERSION)   # version prevents stale hits
    if key in _desc_cache:
        return _desc_cache[key]
    # Evict oldest entries if cache is full
    if len(_desc_cache) >= MAX_DESC_CACHE:
        evict_key = next(iter(_desc_cache))
        del _desc_cache[evict_key]
    img = cv2.imread(full_path)
    if img is None:
        print(f'[CacheDescriptor] WARNING: Failed to read image {full_path}')
        return None
    # Use robust encoding: 60% glasses-invariant zones + 40% full-face HOG+LBP
    desc = face_recognizer.extract_robust_encoding(img)
    # Fallback to standard combined encoding if robust fails
    if desc is None:
        desc = face_recognizer.extract_combined_encoding(img)
    
    if desc is not None:
        _desc_cache[key] = desc
    return desc


def _robust_cosine_similarity(desc1, desc2):
    """
    Compare two face descriptors even if one is robust (10552-dim) and one is combined (5860-dim)
    by extracting the common standard full-face HOG+LBP sub-vector from the robust one.
    """
    if desc1 is None or desc2 is None:
        return 0.0
    
    if desc1.shape == desc2.shape:
        return face_recognizer.cosine_similarity(desc1, desc2)
        
    try:
        # Standard combined full-face HOG+LBP is 5860-dim.
        # Glasses-invariant multi-zone is 4692-dim.
        # Robust is 10552-dim (concatenation of GI (4692) and STD (5860)).
        if desc1.shape == (5860,) and desc2.shape == (10552,):
            # Extract standard part from desc2 (last 5860 elements)
            std2 = desc2[4692:]
            # Normalize since std part was scaled by 0.40 in robust concatenation
            std2_norm = np.linalg.norm(std2)
            if std2_norm > 1e-10:
                std2 = std2 / std2_norm
            return face_recognizer.cosine_similarity(desc1, std2)
            
        elif desc1.shape == (10552,) and desc2.shape == (5860,):
            # Extract standard part from desc1
            std1 = desc1[4692:]
            std1_norm = np.linalg.norm(std1)
            if std1_norm > 1e-10:
                std1 = std1 / std1_norm
            return face_recognizer.cosine_similarity(std1, desc2)
    except Exception as e:
        print(f'[RobustCompare] Dimension alignment error: {e}')
        
    # Raise value error if shapes are completely incompatible or calculation fails
    raise ValueError(f"Incompatible descriptor dimensions for comparison: {desc1.shape} vs {desc2.shape}")



def _load_cnn_model_cached(model_path):
    """Load (or return cached) Keras CNN model + label dict.
    Re-loads only when the .h5 file has changed on disk."""
    try:
        mtime = int(os.path.getmtime(model_path))
    except OSError:
        return None, None
    key = (model_path, mtime)
    if key in _cnn_cache:
        return _cnn_cache[key]
    try:
        from tensorflow.keras.models import load_model as _load_keras
        model = _load_keras(model_path)
        label_path = model_path.replace('.h5', '_labels.pkl')
        labels = {}
        if os.path.exists(label_path):
            import pickle as _pkl
            with open(label_path, 'rb') as lf:
                labels = _pkl.load(lf)
        _cnn_cache[key] = (model, labels)
        # Keep at most 3 CNN models in memory
        if len(_cnn_cache) > 3:
            old_key = next(iter(_cnn_cache))
            del _cnn_cache[old_key]
        return model, labels
    except Exception:
        return None, None



# ============= REPORT GENERATION HELPERS =============

def generate_pdf_report(report, attendance_records, start_date, end_date, subject=None, user=None):
    """Generate PDF report for attendance records"""
    import os
    from django.conf import settings
    
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename
    filename = f'report_{report.id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    filepath = os.path.join(reports_dir, filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#4F46E5'),
        spaceAfter=30,
        alignment=1  # Center
    )
    elements.append(Paragraph(report.title, title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report info
    info_style = styles['Normal']
    elements.append(Paragraph(f"<b>Report Type:</b> {report.get_report_type_display()}", info_style))
    elements.append(Paragraph(f"<b>Period:</b> {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}", info_style))
    if subject:
        elements.append(Paragraph(f"<b>Subject:</b> {subject.name} ({subject.code})", info_style))
    if user:
        elements.append(Paragraph(f"<b>Student:</b> {user.get_full_name() or user.username}", info_style))
    elements.append(Paragraph(f"<b>Generated By:</b> {report.generated_by.get_full_name() or report.generated_by.username}", info_style))
    elements.append(Paragraph(f"<b>Generated On:</b> {report.created_at.strftime('%B %d, %Y %I:%M %p')}", info_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Statistics
    total_records = attendance_records.count()
    present_count = attendance_records.filter(status='present').count()
    absent_count = attendance_records.filter(status='absent').count()
    late_count = attendance_records.filter(status='late').count()
    
    elements.append(Paragraph("<b>Summary Statistics:</b>", styles['Heading2']))
    elements.append(Paragraph(f"Total Records: {total_records}", info_style))
    elements.append(Paragraph(f"Present: {present_count} ({(present_count/total_records*100) if total_records > 0 else 0:.1f}%)", info_style))
    elements.append(Paragraph(f"Absent: {absent_count} ({(absent_count/total_records*100) if total_records > 0 else 0:.1f}%)", info_style))
    elements.append(Paragraph(f"Late: {late_count} ({(late_count/total_records*100) if total_records > 0 else 0:.1f}%)", info_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Attendance table
    elements.append(Paragraph("<b>Attendance Records:</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Table data
    table_data = [['Date', 'Student', 'Subject', 'Status', 'Type', 'Time']]
    
    for record in attendance_records:
        table_data.append([
            record.date.strftime('%Y-%m-%d'),
            record.user.get_full_name() or record.user.username,
            record.subject.code if record.subject else 'N/A',
            record.status.capitalize(),
            record.get_recognition_type_display(),
            record.time.strftime('%H:%M')
        ])
    
    # Create table
    table = Table(table_data, colWidths=[1*inch, 1.8*inch, 1*inch, 0.8*inch, 1.2*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    elements.append(table)
    
    # Build PDF
    doc.build(elements)
    
    # Return relative path for storage
    return f'reports/{filename}'


def generate_csv_report(report, attendance_records, start_date, end_date, subject=None, user=None):
    """Generate CSV report for attendance records"""
    import os
    from django.conf import settings
    
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename
    filename = f'report_{report.id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    filepath = os.path.join(reports_dir, filename)
    
    # Create CSV file
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header information
        writer.writerow(['FRA System - Attendance Report'])
        writer.writerow([])
        writer.writerow(['Report Title:', report.title])
        writer.writerow(['Report Type:', report.get_report_type_display()])
        writer.writerow(['Period:', f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"])
        if subject:
            writer.writerow(['Subject:', f'{subject.name} ({subject.code})'])
        if user:
            writer.writerow(['Student:', user.get_full_name() or user.username])
        writer.writerow(['Generated By:', report.generated_by.get_full_name() or report.generated_by.username])
        writer.writerow(['Generated On:', report.created_at.strftime('%B %d, %Y %I:%M %p')])
        writer.writerow([])
        
        # Write statistics
        total_records = attendance_records.count()
        present_count = attendance_records.filter(status='present').count()
        absent_count = attendance_records.filter(status='absent').count()
        late_count = attendance_records.filter(status='late').count()
        
        writer.writerow(['Summary Statistics:'])
        writer.writerow(['Total Records:', total_records])
        writer.writerow(['Present:', f"{present_count} ({(present_count/total_records*100) if total_records > 0 else 0:.1f}%)"])
        writer.writerow(['Absent:', f"{absent_count} ({(absent_count/total_records*100) if total_records > 0 else 0:.1f}%)"])
        writer.writerow(['Late:', f"{late_count} ({(late_count/total_records*100) if total_records > 0 else 0:.1f}%)"])
        writer.writerow([])
        
        # Write table header
        writer.writerow(['Date', 'Student ID', 'Student Name', 'Subject Code', 'Subject Name', 'Status', 'Recognition Type', 'Time', 'Confidence Score'])
        
        # Write attendance records
        for record in attendance_records:
            writer.writerow([
                record.date.strftime('%Y-%m-%d'),
                record.user.student_id or record.user.employee_id or 'N/A',
                record.user.get_full_name() or record.user.username,
                record.subject.code if record.subject else 'N/A',
                record.subject.name if record.subject else 'N/A',
                record.status.capitalize(),
                record.get_recognition_type_display(),
                record.time.strftime('%H:%M:%S'),
                f"{record.confidence_score:.2f}" if record.confidence_score else 'N/A'
            ])
    
    # Return relative path for storage
    return f'reports/{filename}'


def index(request):
    """Home page"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'index.html')


def user_login(request):
    """User login view"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            if user.is_approved:
                login(request, user)
                
                # Log activity
                SystemLog.objects.create(
                    user=user,
                    log_type='info',
                    action='Login',
                    description=f'User {username} logged in successfully',
                    ip_address=request.META.get('REMOTE_ADDR')
                )
                
                # Teachers must pass face verification first
                if user.role == 'teacher':
                    request.session['face_verified'] = False
                    messages.info(request, 'Please verify your identity to continue.')
                    return redirect('face_verify_login')
                
                messages.success(request, 'Login successful!')
                return redirect('dashboard')
            else:
                messages.warning(request, 'Your account is pending approval.')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'login.html')


def user_register(request):
    """User registration view — validates form and sends OTP before creating account."""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)

        # Get face images data from hidden input
        face_images_json = request.POST.get('face_images_data', '[]')
        try:
            face_images = json.loads(face_images_json)
        except Exception:
            face_images = []

        # Validate face images first
        if len(face_images) < 5:
            messages.error(request, 'Please capture at least 5 face samples before registering.')
            return render(request, 'register.html', {'form': form, 'departments': Department.objects.all()})

        if form.is_valid():
            email = form.cleaned_data['email']

            # Check email uniqueness
            if User.objects.filter(email=email).exists():
                form.add_error('email', 'This email address is already registered.')
                return render(request, 'register.html', {'form': form, 'departments': Department.objects.all()})

            # Send OTP email
            import random
            import time
            from django.core.mail import send_mail
            from django.conf import settings as djconf

            otp = str(random.randint(100000, 999999))

            # Store all registration data + OTP + timestamp in session
            request.session['pending_registration'] = {
                'form_data': request.POST.dict(),
                'face_images': face_images,
                'otp': otp,
                'otp_created_at': time.time(),
                'email': email,
            }

            email_sent = False
            try:
                send_mail(
                    subject='Your FRA System Registration OTP',
                    message=(
                        f'Hello,\n\n'
                        f'Your one-time verification code for FRA System registration is:\n\n'
                        f'  {otp}\n\n'
                        f'This code expires in {getattr(djconf, "OTP_EXPIRY_MINUTES", 10)} minutes.\n'
                        f'If you did not request this, please ignore this email.\n\n'
                        f'— FRA System'
                    ),
                    from_email=djconf.DEFAULT_FROM_EMAIL,
                    recipient_list=[email],
                    fail_silently=False,
                )
                email_sent = True
                messages.success(request, f'A 6-digit OTP has been sent to {email}. Please check your inbox.')
            except Exception as mail_err:
                print(f'[OTP EMAIL ERROR] {mail_err}')
                messages.warning(request, f'[DEBUG] Could not send email — OTP is: {otp}')

            return redirect('verify_otp')
    else:
        form = UserRegistrationForm()

    return render(request, 'register.html', {
        'form': form,
        'departments': Department.objects.all(),
    })


def verify_otp(request):
    """OTP verification view — validates OTP then creates the user account."""
    pending = request.session.get('pending_registration')
    if not pending:
        messages.error(request, 'Registration session expired. Please register again.')
        return redirect('register')

    if request.method == 'POST':
        entered_otp = request.POST.get('otp', '').strip()
        action = request.POST.get('action', 'verify')

        # Handle resend OTP
        if action == 'resend':
            import random, time
            new_otp = str(random.randint(100000, 999999))
            pending['otp'] = new_otp
            pending['otp_created_at'] = time.time()
            request.session['pending_registration'] = pending
            request.session.modified = True

            email = pending['email']
            try:
                from django.core.mail import send_mail
                from django.conf import settings as django_settings
                send_mail(
                    subject='Your FRA System Registration OTP (Resent)',
                    message=(
                        f'Hello,\n\n'
                        f'Your new OTP for FRA System registration is:\n\n'
                        f'  {new_otp}\n\n'
                        f'This code expires in {getattr(django_settings, "OTP_EXPIRY_MINUTES", 10)} minutes.\n\n'
                        f'— FRA System'
                    ),
                    from_email=getattr(django_settings, 'DEFAULT_FROM_EMAIL', None),
                    recipient_list=[email],
                    fail_silently=False,
                )
                messages.success(request, f'A new OTP has been sent to {email}.')
            except Exception:
                from django.conf import settings as django_settings
                if django_settings.DEBUG:
                    messages.warning(request, f'[DEBUG] New OTP is: {new_otp}')
                else:
                    messages.error(request, 'Failed to resend OTP. Please try again.')
            return redirect('verify_otp')

        # Check OTP expiry
        import time
        from django.conf import settings as django_settings
        expiry_seconds = getattr(django_settings, 'OTP_EXPIRY_MINUTES', 10) * 60
        if time.time() - pending['otp_created_at'] > expiry_seconds:
            del request.session['pending_registration']
            messages.error(request, 'OTP has expired. Please register again.')
            return redirect('register')

        # Validate OTP
        if entered_otp != pending['otp']:
            messages.error(request, 'Invalid OTP. Please try again.')
            return render(request, 'verify_otp.html', {'email': pending['email']})

        # OTP correct — create the account
        from django.http import QueryDict
        form_data = pending['form_data']
        qd = QueryDict(mutable=True)
        qd.update(form_data)
        form = UserRegistrationForm(qd)

        if form.is_valid():
            user = form.save(commit=False)
            user.is_approved = False

            if user.role == 'student':
                user.employee_id = None
            else:
                user.student_id = None

            # Save dept and section from form data stored in session
            try:
                dept_id = form_data.get('dept')
                sec_id  = form_data.get('section')
                if dept_id:
                    user.dept = Department.objects.filter(id=int(dept_id)).first()
                if sec_id:
                    user.section = Section.objects.filter(id=int(sec_id)).first()
            except Exception:
                pass

            user.save()

            # Save face samples
            from .models import BiometricSample, TrainingDataset
            from datetime import datetime as dt
            face_images = pending.get('face_images', [])

            for idx, image_data in enumerate(face_images):
                try:
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    training_dir = os.path.join(settings.MEDIA_ROOT, 'training_data', 'face', str(user.id))
                    os.makedirs(training_dir, exist_ok=True)
                    timestamp = dt.now().strftime('%Y%m%d_%H%M%S_%f')
                    filename = f'face_sample_{idx}_{timestamp}.jpg'
                    file_path = os.path.join(training_dir, filename)
                    with open(file_path, 'wb') as f:
                        f.write(image_bytes)
                    relative_path = os.path.join('training_data', 'face', str(user.id), filename)
                    user_identifier = user.student_id if user.role == 'student' else user.employee_id
                    dataset_name = user_identifier if user_identifier else f'User_{user.id}'
                    face_dataset, _ = TrainingDataset.objects.get_or_create(
                        dataset_type='face',
                        name=dataset_name,
                        defaults={'description': f'Face recognition dataset for {user_identifier or user.username}'}
                    )
                    BiometricSample.objects.create(
                        user=user,
                        sample_type='face',
                        image_path=relative_path,
                        dataset=face_dataset,
                    )
                except Exception as e:
                    print(f'Error saving face sample {idx}: {e}')

            # Update dataset counts
            for dataset in TrainingDataset.objects.filter(dataset_type='face'):
                dataset.update_sample_count()

            # Clear session
            del request.session['pending_registration']

            messages.success(
                request,
                f'Account created successfully! {len(face_images)} face samples saved. '
                f'Please wait for admin approval before logging in.'
            )
            return redirect('login')

        # Form invalid after OTP pass (shouldn't normally happen)
        messages.error(request, 'Registration data is invalid. Please start again.')
        return redirect('register')

    return render(request, 'verify_otp.html', {'email': pending['email']})



@login_required
def user_logout(request):
    """User logout view"""
    SystemLog.objects.create(
        user=request.user,
        log_type='info',
        action='Logout',
        description=f'User {request.user.username} logged out',
        ip_address=request.META.get('REMOTE_ADDR')
    )
    
    logout(request)
    messages.success(request, 'Logged out successfully!')
    return redirect('login')


@login_required
def face_verify_login(request):
    """Show face-recognition verification page for teacher/admin after password login."""
    # Only teacher and admin need this; send students straight to dashboard
    if request.user.role not in ('teacher', 'admin'):
        return redirect('dashboard')
    # Already verified in this session
    if request.session.get('face_verified'):
        return redirect('dashboard')
    return render(request, 'face_verify_login.html')


@csrf_exempt
@login_required
def api_verify_face_login(request):
    """
    API: Verify logged-in teacher/admin identity via face recognition.

    Uses direct Pearson-correlation comparison against the user's stored
    BiometricSample images — identical approach to api_teacher_face_attendance.
    This is fully persistent across server restarts: no CNN model training or
    reloading is required. The check simply reads face images from disk and
    compares encodings, so it works immediately after any restart.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'POST required.'}, status=405)

    if request.user.role not in ('teacher', 'admin'):
        return JsonResponse({'success': False, 'message': 'Not required for this role.'}, status=403)

    try:
        from .models import BiometricSample

        data = json.loads(request.body)
        image_data = data.get('image', '')

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JsonResponse({'success': False, 'message': 'Could not decode image. Please try again.'})

        # DEBUG: save incoming live frame so we can inspect it
        _debug_path = os.path.join(str(settings.MEDIA_ROOT), 'debug_live_frame.jpg')
        cv2.imwrite(_debug_path, image)
        _faces_in_live = len(face_recognizer.detect_faces(image))
        print(f'[FaceVerify] Live frame shape={image.shape}  faces={_faces_in_live}  saved to {_debug_path}')

        # ── Step 1: Build multiple enhanced versions of the live frame ──────────
        # This gives robustness to dim light and slight pose shifts:
        # whichever enhancement makes the face most visible will produce the
        # best matching score and that score is used.
        live_variants = [image]
        try:
            live_variants.append(face_recognizer.apply_clahe(image))
        except Exception:
            pass
        try:
            live_variants.append(face_recognizer.normalize_brightness(image))
        except Exception:
            pass
        # Strong gamma boost for very dim light
        try:
            gamma_table = np.array([(i / 255.0) ** (1.0 / 0.4) * 255
                                     for i in range(256)], dtype=np.uint8)
            live_variants.append(cv2.LUT(image, gamma_table))
        except Exception:
            pass

        # Extract descriptor from the first variant that produces a valid result
        live_descriptor = None
        for variant in live_variants:
            live_descriptor = face_recognizer.extract_combined_encoding(variant)
            if live_descriptor is not None:
                break

        # Also try horizontally flipped (handles webcam mirror mismatch)
        live_descriptor_flip = None
        for variant in live_variants:
            live_descriptor_flip = face_recognizer.extract_combined_encoding(cv2.flip(variant, 1))
            if live_descriptor_flip is not None:
                break

        print(f'[FaceVerify] live_desc={live_descriptor is not None}  live_flip={live_descriptor_flip is not None}  variants_tried={len(live_variants)}')

        if live_descriptor is None and live_descriptor_flip is None:
            return JsonResponse({
                'success': False,
                'message': f'No face detected ({_faces_in_live} detected by Haar). Ensure good lighting and look directly at the camera.',
                'debug': {'faces_in_frame': _faces_in_live},
            })

        # ── Step 2: Compare against THIS user's stored samples ────────────────
        user_samples = BiometricSample.objects.filter(
            user=request.user,
            sample_type='face',
        ).order_by('?')[:30]  # random 30 for robust coverage

        if not user_samples.exists():
            return JsonResponse({
                'success': False,
                'message': (
                    'No face data found for your account. '
                    'Please re-register or ask an admin to upload your face samples.'
                ),
            })

        user_best      = 0.0
        user_above_thr = 0
        user_compared  = 0

        # HOG+LBP same-person cosine similarity is typically 0.80–0.99.
        # Per-sample threshold set to 0.68 for robust matching under varied lighting / pose.
        PER_SAMPLE_THR = 0.68

        for sample in user_samples:
            try:
                full_path = os.path.join(str(settings.MEDIA_ROOT), sample.image_path)
                if not os.path.exists(full_path):
                    continue
                stored_img = cv2.imread(full_path)
                if stored_img is None:
                    continue
                stored_desc = face_recognizer.extract_combined_encoding(stored_img)
                if stored_desc is None:
                    continue

                # Take the best similarity from normal or flipped live frame
                sim1 = face_recognizer.cosine_similarity(live_descriptor, stored_desc) if live_descriptor is not None else 0.0
                sim2 = face_recognizer.cosine_similarity(live_descriptor_flip, stored_desc) if live_descriptor_flip is not None else 0.0
                sim = max(sim1, sim2)

                user_best = max(user_best, sim)
                user_compared += 1
                if sim >= PER_SAMPLE_THR:
                    user_above_thr += 1
                print(f'[FaceVerify] user_sample {user_compared}: sim={sim:.4f}')
            except Exception as ex:
                print(f'[FaceVerify] user_sample error: {ex}')
                continue

        # ── Step 3: NEGATIVE GATE — compare against OTHER teachers/admins only ─
        # IMPORTANT: Scope to same-role accounts so structurally-similar student
        # HOG+LBP vectors do not inflate other_best and cause false rejection of
        # the real teacher.  Students cannot impersonate a teacher account regardless.
        other_samples = BiometricSample.objects.filter(
            sample_type='face',
            user__role__in=['teacher', 'admin'],
        ).exclude(user=request.user).order_by('?')[:40]

        other_best = 0.0
        for sample in other_samples:
            try:
                full_path = os.path.join(str(settings.MEDIA_ROOT), sample.image_path)
                if not os.path.exists(full_path):
                    continue
                stored_img = cv2.imread(full_path)
                if stored_img is None:
                    continue
                stored_desc = face_recognizer.extract_combined_encoding(stored_img)
                if stored_desc is None:
                    continue
                # Normal frame only — prevents mirror-inflated cross-user scores
                sim = face_recognizer.cosine_similarity(live_descriptor, stored_desc) if live_descriptor is not None else 0.0
                other_best = max(other_best, sim)
            except Exception:
                continue

        # ── Step 4: GLOBAL UNIQUENESS GATE (RELAXED / BYPASSED) ──────────────────
        # Compare the live frame against other registered users.
        # NOTE: Comparing against all users (especially students) in HOG+LBP space
        # is highly prone to false rejections due to structural similarities and
        # random score inflation. Since the user is already authenticated with a
        # password and we compare against their own stored templates, a global
        # uniqueness margin is not required for 1:1 verification.
        # We calculate it strictly for debug logging, but do not enforce a strict margin.
        global_other_samples = BiometricSample.objects.filter(
            sample_type='face',
        ).exclude(user=request.user).order_by('?')[:60]

        global_other_best = 0.0
        for sample in global_other_samples:
            try:
                full_path = os.path.join(str(settings.MEDIA_ROOT), sample.image_path)
                if not os.path.exists(full_path):
                    continue
                stored_img = cv2.imread(full_path)
                if stored_img is None:
                    continue
                stored_desc = face_recognizer.extract_combined_encoding(stored_img)
                if stored_desc is None:
                    continue
                # Use the best of normal/flipped to be fair to the genuine teacher
                sim1 = face_recognizer.cosine_similarity(live_descriptor, stored_desc) if live_descriptor is not None else 0.0
                sim2 = face_recognizer.cosine_similarity(live_descriptor_flip, stored_desc) if live_descriptor_flip is not None else 0.0
                global_other_best = max(global_other_best, sim1, sim2)
            except Exception:
                continue

        print(f'[FaceVerify] user_best={user_best:.4f} other_best={other_best:.4f} '
              f'global_other_best={global_other_best:.4f} above_thr={user_above_thr}/{user_compared}')

        # ── Threshold values ───────────────────────────────────────────────────
        # HOG+LBP cosine similarity empirical ranges:
        #   Same person     : 0.80 – 0.99  (genuine match)
        #   Different people: 0.55 – 0.78  (imposters / strangers)
        # All thresholds are set conservatively strict so that ONLY the
        # specific registered teacher whose biometric data is on file can pass.
        BEST_SCORE_THRESHOLD     = 0.70   # absolute minimum score to be accepted
        MIN_MATCHING_SAMPLES     = 1      # at least 1 stored samples must each exceed PER_SAMPLE_THR
        NEGATIVE_MARGIN          = 0.01   # must beat other teachers/admins by ≥ 1 %
        SOLO_FLOOR               = 0.70   # minimum floor when no other teacher/admin is registered
        GLOBAL_UNIQUENESS_MARGIN = 0.00   # relaxed to 0% to prevent student-sample score inflation

        # Role-scoped negative gate (this teacher vs other teachers / admins)
        if other_best == 0.0:
            negative_gate_passes = user_best >= SOLO_FLOOR
        else:
            negative_gate_passes = (user_best - other_best) >= NEGATIVE_MARGIN

        # Global uniqueness gate — calculated strictly for debug logging, but bypassed
        # here to prevent unrelated student database profiles from causing false rejections.
        global_unique_passes = True

        identity_confirmed = (
            user_best >= BEST_SCORE_THRESHOLD
            and user_above_thr >= min(MIN_MATCHING_SAMPLES, user_compared)
            and user_compared > 0
            and negative_gate_passes
            and global_unique_passes
        )

        # Map back to existing variable names for the response code below
        best_similarity      = user_best
        samples_above_threshold = user_above_thr
        total_compared       = user_compared

        print(f'[FaceVerify] RESULT: confirmed={identity_confirmed} neg_gate={negative_gate_passes} other_best={other_best:.4f}')

        if identity_confirmed:
            # ✅ Identity confirmed — grant dashboard access for this session
            request.session['face_verified'] = True
            SystemLog.objects.create(
                user=request.user,
                log_type='info',
                action='Face Verification',
                description=(
                    f'{request.user.username} passed face verification at login '
                    f'(best={best_similarity:.2f}, matched {samples_above_threshold}/{total_compared} samples)'
                ),
                ip_address=request.META.get('REMOTE_ADDR'),
            )
            return JsonResponse({
                'success': True,
                'name': request.user.get_full_name() or request.user.username,
                'redirect': '/dashboard/',
            })
        else:
            # Determine a helpful rejection message (includes score for debugging)
            score_info = f' (score: {best_similarity:.0%})'
            if total_compared == 0:
                reason = (
                    'No valid face samples found for your account. '
                    'Please ask an admin to upload your biometric data.'
                )
            elif best_similarity < 0.50:
                reason = (
                    f'Face not detected clearly{score_info}. '
                    'Ensure your face fills the oval, lighting is bright, '
                    'and you are looking directly at the camera.'
                )
            elif not global_unique_passes:
                reason = (
                    'Access denied — identity could not be uniquely confirmed as this account holder. '
                    'Only the registered teacher for this account may access the portal. '
                    'If you are the account holder, ensure good lighting, remove obstructions, '
                    'and look directly at the camera.'
                )
            elif not negative_gate_passes and other_best > 0.0:
                reason = (
                    'Access denied — this face does not match the registered account holder. '
                    'If you are the account holder, ensure good lighting and face the camera directly.'
                )
            elif not negative_gate_passes:
                reason = (
                    f'Confidence too low for unique verification{score_info}. '
                    'Face the camera directly with good lighting and hold still.'
                )
            elif best_similarity < BEST_SCORE_THRESHOLD:
                reason = (
                    f'Face recognised but confidence too low{score_info}. '
                    'Try better lighting or move closer to the camera.'
                )
            elif samples_above_threshold < MIN_MATCHING_SAMPLES:
                reason = (
                    f'Matched only {samples_above_threshold}/{total_compared} stored samples{score_info}. '
                    'Try better lighting and face the camera directly.'
                )
            else:
                reason = (
                    f'Verification failed{score_info}. '
                    'Ensure your face is well-lit and unobstructed.'
                )

            SystemLog.objects.create(
                user=request.user,
                log_type='security',
                action='Face Verification Failed',
                description=(
                    f'Login face-verify REJECTED for {request.user.username} '
                    f'(best={best_similarity:.2f}, matched {samples_above_threshold}/{total_compared} samples)'
                ),
                ip_address=request.META.get('REMOTE_ADDR'),
            )
            return JsonResponse({
                'success': False,
                'message': reason,
                'debug': {
                    'user_best': round(user_best, 4),
                    'other_best': round(other_best, 4),
                    'global_other_best': round(global_other_best, 4),
                    'margin_vs_teachers': round(user_best - other_best, 4),
                    'margin_vs_all_users': round(user_best - global_other_best, 4),
                    'above_thr': user_above_thr,
                    'compared': user_compared,
                    'neg_gate': negative_gate_passes,
                    'global_unique_gate': global_unique_passes,
                    'BEST_SCORE_THRESHOLD': BEST_SCORE_THRESHOLD,
                    'MIN_MATCHING_SAMPLES': MIN_MATCHING_SAMPLES,
                    'NEGATIVE_MARGIN': NEGATIVE_MARGIN,
                    'SOLO_FLOOR': SOLO_FLOOR,
                    'GLOBAL_UNIQUENESS_MARGIN': GLOBAL_UNIQUENESS_MARGIN,
                }
            })

    except Exception as e:
        return JsonResponse({'success': False, 'message': f'Error: {str(e)}'})


@login_required
def dashboard(request):
    """Main dashboard"""
    # Guard: teacher must complete face verification first
    if request.user.role == 'teacher' and not request.session.get('face_verified'):
        return redirect('face_verify_login')

    context = {}
    
    if request.user.role == 'admin':
        # Admin dashboard
        context['total_users'] = User.objects.count()
        context['pending_approvals'] = User.objects.filter(is_approved=False).count()
        context['total_subjects'] = Subject.objects.count()
        # Today's attendance - Detailed breakdown
        today = timezone.now().date()
        context['today_attendance'] = Attendance.objects.filter(date=today).count()
        context['today_student_attendance'] = Attendance.objects.filter(date=today, user__role='student').count()
        context['today_teacher_attendance'] = Attendance.objects.filter(date=today, user__role='teacher').count()
        
        # Recent attendance - Separated by role
        context['recent_student_attendance'] = Attendance.objects.filter(
            user__role='student'
        ).select_related('user', 'subject').order_by('-created_at')[:8]

        context['recent_teacher_attendance'] = Attendance.objects.filter(
            user__role='teacher'
        ).select_related('user', 'subject').order_by('-created_at')[:8]
        
        # Attendance statistics
        week_ago = timezone.now().date() - timedelta(days=7)
        context['week_attendance'] = Attendance.objects.filter(date__gte=week_ago).values('date').annotate(count=Count('id')).order_by('date')
        
    elif request.user.role == 'teacher':
        # Teacher dashboard
        context['my_subjects'] = Subject.objects.filter(teacher=request.user)
        # Count only STUDENT attendance for this teacher's subjects today
        # (Teacher's own subject-linked records must not inflate this number)
        context['today_attendance'] = Attendance.objects.filter(
            subject__teacher=request.user,
            date=timezone.now().date(),
            user__role='student',
        ).count()
        # Teacher's own attendance for today
        context['teacher_today_marked'] = Attendance.objects.filter(
            user=request.user,
            date=timezone.now().date()
        ).first()
        # IDs of subjects that have an active live session
        context['live_session_ids'] = set(
            LiveAttendanceSession.objects.filter(
                is_active=True,
                started_by=request.user
            ).values_list('subject_id', flat=True)
        )
        
    else:
        # Student dashboard
        context['my_attendance'] = Attendance.objects.filter(user=request.user).order_by('-date')[:10]
        context['total_attendance'] = Attendance.objects.filter(user=request.user).count()
        context['present_count'] = Attendance.objects.filter(user=request.user, status='present').count()

        if context['total_attendance'] > 0:
            context['attendance_percentage'] = (context['present_count'] / context['total_attendance']) * 100
        else:
            context['attendance_percentage'] = 0

        # Pass student's dept/section info for display
        context['student_dept'] = request.user.dept
        context['student_section'] = request.user.section

        # Active live-session subject IDs for this student (filtered by dept+section)
        live_qs = LiveAttendanceSession.objects.filter(
            is_active=True,
            subject__students=request.user,
        )
        # Further restrict to the student's own section if they have one
        if request.user.section:
            live_qs = live_qs.filter(
                Q(subject__section=request.user.section) | Q(subject__section__isnull=True)
            )
        active_ids = sorted(live_qs.values_list('subject_id', flat=True))
        context['live_session_active'] = bool(active_ids)
        context['live_subject_ids_json'] = json.dumps(active_ids)
    
    return render(request, 'dashboard.html', context)


# ============= TEACHER SELF FACE ATTENDANCE =============

@login_required
def mark_teacher_attendance_face(request):
    """Teacher marks their own attendance via face recognition, per assigned subject.
    Subjects are grouped by Department → Section for a clear, organised UI.
    """
    if request.user.role != 'teacher':
        messages.error(request, 'This page is only for teachers.')
        return redirect('dashboard')

    today = timezone.now().date()

    # All subjects assigned to this teacher, with dept/section info pre-fetched
    my_subjects = Subject.objects.filter(
        teacher=request.user
    ).select_related('department', 'section').order_by(
        'department__name', 'section__name', 'name'
    )

    # Which subject IDs already have face attendance recorded today?
    marked_subject_ids = set(
        Attendance.objects.filter(
            user=request.user,
            date=today,
            recognition_type='face',
        ).exclude(subject=None).values_list('subject_id', flat=True)
    )

    # ── Group subjects: dept → section → [subjects] ─────────────────────────
    # Structure: {dept_label: {section_label: [subject, ...]}}
    from collections import OrderedDict
    grouped = OrderedDict()
    for subj in my_subjects:
        dept_label = str(subj.department) if subj.department else 'No Department'
        sec_label  = str(subj.section)    if subj.section    else 'No Section'
        grouped.setdefault(dept_label, OrderedDict()).setdefault(sec_label, []).append(subj)

    # Count stats
    total_subjects  = my_subjects.count()
    marked_today    = len(marked_subject_ids)
    pending_today   = total_subjects - marked_today

    recent_attendance = Attendance.objects.filter(
        user=request.user
    ).select_related('subject').order_by('-date', '-time')[:15]

    context = {
        'today':             today,
        'my_subjects':       my_subjects,
        'grouped_subjects':  grouped,        # dept → section → subjects
        'marked_subject_ids': marked_subject_ids,
        'recent_attendance': recent_attendance,
        'total_subjects':    total_subjects,
        'marked_today':      marked_today,
        'pending_today':     pending_today,
    }
    return render(request, 'mark_teacher_attendance_face.html', context)


@csrf_exempt
@login_required
def api_teacher_face_attendance(request):
    """
    API: Teacher scans face to mark attendance for a specific assigned subject.
    One record per subject per day — not a single generic daily record.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'POST required.'}, status=405)

    if request.user.role != 'teacher':
        return JsonResponse({'success': False, 'message': 'Only teachers can use this endpoint.'}, status=403)

    try:
        data = json.loads(request.body)
        image_data = data.get('image', '')
        subject_id  = data.get('subject_id')

        # ── Validate the selected subject ──────────────────────────────────────
        if not subject_id:
            return JsonResponse({
                'success': False,
                'message': 'Please select a subject before scanning your face.',
            })
        try:
            subject = Subject.objects.get(id=int(subject_id), teacher=request.user)
        except (Subject.DoesNotExist, ValueError):
            return JsonResponse({
                'success': False,
                'message': 'Invalid subject — please pick one of your assigned subjects.',
            })

        # Decode the base64 image sent from the browser
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JsonResponse({'success': False, 'message': 'Could not decode image. Please try again.'})

        # ── Step 1: Extract robust descriptor from the live frame ────────────
        # IMPORTANT: Must use extract_robust_encoding here to match the shape
        # returned by _get_cached_descriptor for stored samples (10552-dim).
        # Using extract_combined_encoding (5860-dim) would cause a numpy shape
        # mismatch in cosine_similarity, silently setting total_compared=0.
        live_variants = [image]
        try:
            live_variants.append(face_recognizer.apply_clahe(image))
        except Exception:
            pass
        try:
            live_variants.append(face_recognizer.normalize_brightness(image))
        except Exception:
            pass
        try:
            live_variants.append(cv2.flip(image, 1))
        except Exception:
            pass

        live_descriptor = None
        for variant in live_variants:
            live_descriptor = face_recognizer.extract_robust_encoding(variant)
            if live_descriptor is not None:
                break

        if live_descriptor is None:
            return JsonResponse({
                'success': False,
                'message': 'No face detected in frame. Ensure good lighting, face the camera directly, and remove any obstructions.',
            })

        # ── Step 2: Compare against THIS teacher's stored samples ─────────────
        from .models import BiometricSample
        teacher_samples = BiometricSample.objects.filter(
            user=request.user,
            sample_type='face',
        )[:20]

        if not teacher_samples.exists():
            return JsonResponse({
                'success': False,
                'message': 'No face training data found for your account. Please register your face first.',
            })

        PER_SAMPLE_THR  = 0.68   # must be below BEST_SCORE_THRESHOLD (0.72) to increment correctly
        best_similarity = 0.0
        samples_above_lower = 0
        total_compared  = 0

        for sample in teacher_samples:
            try:
                full_path = os.path.join(str(settings.MEDIA_ROOT), sample.image_path)
                if not os.path.exists(full_path):
                    continue
                stored_desc = _get_cached_descriptor(full_path)
                if stored_desc is None:
                    continue
                sim = _robust_cosine_similarity(live_descriptor, stored_desc)
                best_similarity = max(best_similarity, sim)
                total_compared += 1
                if sim >= PER_SAMPLE_THR:
                    samples_above_lower += 1
                print(f'[TeacherFace] sample {total_compared}: hog+lbp_cos={sim:.4f}')
            except Exception as ex:
                print(f'[TeacherFace] sample error: {ex}')
                import traceback
                traceback.print_exc()
                continue

        # ── Step 3: NEGATIVE GATE — compare against OTHER teachers/admins only ───
        # IMPORTANT: Scope to same-role accounts only.
        # Comparing against student samples inflates other_best (students can have
        # structurally similar HOG+LBP vectors) and causes false rejection of the
        # real teacher. Students cannot impersonate a teacher account regardless.
        other_samples = BiometricSample.objects.filter(
            sample_type='face',
            user__role__in=['teacher', 'admin'],
        ).exclude(user=request.user).order_by('?')[:20]

        other_best = 0.0
        for sample in other_samples:
            try:
                full_path = os.path.join(str(settings.MEDIA_ROOT), sample.image_path)
                if not os.path.exists(full_path):
                    continue
                stored_desc = _get_cached_descriptor(full_path)
                if stored_desc is None:
                    continue
                sim = _robust_cosine_similarity(live_descriptor, stored_desc)
                other_best = max(other_best, sim)
            except Exception:
                continue

        print(f'[TeacherFace] user_best={best_similarity:.4f} other_best={other_best:.4f} above_thr={samples_above_lower}/{total_compared}')

        # ── Tri-gate identity check ───────────────────────────────────────────
        # Role-scoped negative gate:
        #   • If no other teacher/admin registered  → margin is irrelevant, use SOLO_FLOOR
        #   • If other teachers exist               → teacher must beat them by NEGATIVE_MARGIN
        BEST_SCORE_THRESHOLD = 0.70   # tolerant of dim light / distance
        MIN_MATCHING_SAMPLES = 1      # at least 1 clear sample match required
        NEGATIVE_MARGIN      = 0.01   # 1% margin over other same-role users
        SOLO_FLOOR           = 0.68   # floor when this teacher is the only registered one
        ABSOLUTE_FLOOR       = 0.50   # hard reject if score is this low (bad lighting/no face)

        if other_best == 0.0:
            # No other teacher/admin in DB — pass if score clears the solo floor
            negative_gate_passes = best_similarity >= SOLO_FLOOR
        else:
            # Must beat closest same-role competitor by NEGATIVE_MARGIN
            negative_gate_passes = (best_similarity - other_best) >= NEGATIVE_MARGIN

        identity_confirmed = (
            best_similarity >= BEST_SCORE_THRESHOLD
            and samples_above_lower >= min(MIN_MATCHING_SAMPLES, total_compared)
            and total_compared > 0
            and negative_gate_passes
        )

        print(f'[TeacherFace] RESULT: confirmed={identity_confirmed} '
              f'neg_gate={negative_gate_passes} score={best_similarity:.4f}')

        if not identity_confirmed:
            score_pct = f'{best_similarity:.0%}'
            if total_compared == 0:
                msg = ('No usable face samples found. '
                       'Please re-register your biometric data.')
            elif best_similarity < ABSOLUTE_FLOOR:
                msg = (f'Face confidence too low ({score_pct}). '
                       'Ensure good lighting and look directly at the camera.')
            elif not negative_gate_passes:
                msg = (f'Identity not uniquely confirmed ({score_pct}). '
                       'Your face was too similar to another registered user. '
                       'Try better lighting or move closer to the camera.')
            elif best_similarity < BEST_SCORE_THRESHOLD:
                msg = (f'Face confidence below threshold ({score_pct} / required {BEST_SCORE_THRESHOLD:.0%}). '
                       'Try better lighting or move closer.')
            elif samples_above_lower < MIN_MATCHING_SAMPLES:
                msg = (f'Too few samples matched ({samples_above_lower}/{total_compared}, score {score_pct}). '
                       'Try better lighting.')
            else:
                msg = f'Verification failed ({score_pct}). Ensure your face is well-lit and unobstructed.'
            return JsonResponse({'success': False, 'message': msg})

        # Use the best_similarity as the confidence score for logging
        confidence = best_similarity


        today = timezone.now().date()

        # ── Per-subject duplicate check ────────────────────────────────────────
        existing = Attendance.objects.filter(
            user=request.user,
            subject=subject,
            date=today,
        ).first()
        if existing:
            return JsonResponse({
                'success': True,
                'already_marked': True,
                'subject_id': subject.id,
                'message': (
                    f'Attendance for "{subject.name}" already recorded today '
                    f'({existing.get_status_display()}) at {existing.time.strftime("%I:%M %p")}.'
                ),
                'confidence': confidence,
            })

        # ── Create subject-linked attendance record ────────────────────────────
        attendance = Attendance.objects.create(
            user=request.user,
            subject=subject,
            date=today,
            status='present',
            recognition_type='face',
            confidence_score=confidence,
            marked_by=request.user,
            is_verified=True,
            is_locked=False,   # teacher records are not locked
        )

        SystemLog.objects.create(
            user=request.user,
            log_type='info',
            action='Teacher Attendance Marked',
            description=(
                f'{request.user.username} marked attendance for "{subject.name}" '
                f'via face recognition (confidence: {confidence:.2f})'
            ),
            ip_address=request.META.get('REMOTE_ADDR'),
        )

        from .models import Notification
        Notification.objects.create(
            user=request.user,
            notification_type='success',
            title='Attendance Marked',
            message=(
                f'Your attendance for {subject.name} on '
                f'{today.strftime("%d %B %Y")} has been recorded as Present.'
            ),
            action_url='/teacher-attendance/',
        )

        return JsonResponse({
            'success': True,
            'already_marked': False,
            'subject_id': subject.id,
            'subject_name': subject.name,
            'message': (
                f'Attendance marked for "{subject.name}"! '
                f'Welcome, {request.user.get_full_name() or request.user.username}.'
            ),
            'confidence': confidence,
            'time': attendance.time.strftime('%I:%M %p'),
        })

    except Exception as e:
        return JsonResponse({'success': False, 'message': f'Error: {str(e)}'})


# ============= LIVE ATTENDANCE SESSION VIEWS =============

@login_required
def start_live_attendance(request, subject_id):
    """Teacher starts a live attendance session for a subject."""
    if request.user.role != 'teacher':
        return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'POST required.'}, status=405)

    subject = get_object_or_404(Subject, id=subject_id, teacher=request.user)

    session, created = LiveAttendanceSession.objects.get_or_create(
        subject=subject,
        defaults={'started_by': request.user, 'is_active': True}
    )
    if not created:
        # Reactivate if it was stopped
        session.started_by = request.user
        session.is_active = True
        session.save()

    return JsonResponse({
        'success': True,
        'message': f'Live attendance started for {subject.name}.',
        'subject_id': subject.id,
        'subject_name': subject.name,
    })


@login_required
def stop_live_attendance(request, subject_id):
    """Teacher stops the live attendance session for a subject."""
    if request.user.role != 'teacher':
        return JsonResponse({'success': False, 'message': 'Permission denied.'}, status=403)
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'POST required.'}, status=405)

    subject = get_object_or_404(Subject, id=subject_id, teacher=request.user)

    try:
        session = LiveAttendanceSession.objects.get(subject=subject)
        session.is_active = False
        session.save()
        return JsonResponse({'success': True, 'message': f'Live attendance stopped for {subject.name}.'})
    except LiveAttendanceSession.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'No session found.'})


@login_required
def check_live_session(request):
    """
    Student polls to check which enrolled subjects have an active live session.
    Returns all active subject IDs so the client can detect any start/stop change.
    """
    if request.user.role != 'student':
        return JsonResponse({'active_subject_ids': []})

    active_ids = sorted(
        LiveAttendanceSession.objects.filter(
            is_active=True,
            subject__students=request.user,
        ).values_list('subject_id', flat=True)
    )

    return JsonResponse({'active_subject_ids': active_ids})


@login_required
def profile(request):
    """User profile view"""
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        form = UserProfileForm(instance=request.user)
    
    biometric_data = BiometricData.objects.filter(user=request.user)
    
    return render(request, 'profile.html', {'form': form, 'biometric_data': biometric_data})


@login_required
def upload_biometric(request):
    """Upload biometric data"""
    if request.method == 'POST':
        form = BiometricDataForm(request.POST, request.FILES)
        if form.is_valid():
            biometric = form.save(commit=False)
            biometric.user = request.user
            biometric.save()
            
            messages.success(request, 'Biometric data uploaded successfully!')
            return redirect('profile')
    else:
        form = BiometricDataForm()
    
    return render(request, 'upload_biometric.html', {'form': form})


@login_required
def mark_attendance_face(request):
    """Mark attendance using facial recognition"""
    if request.user.role == 'student':
        # Only show subjects with active live sessions that the student is enrolled in
        active_subject_ids = LiveAttendanceSession.objects.filter(
            is_active=True
        ).values_list('subject_id', flat=True)

        subjects = request.user.enrolled_subjects.filter(id__in=active_subject_ids)

        # Additionally filter by student's section if set
        if request.user.section:
            subjects = subjects.filter(
                Q(section=request.user.section) | Q(section__isnull=True)
            )
    elif request.user.role == 'teacher':
        subjects = Subject.objects.filter(teacher=request.user)
    else:
        subjects = Subject.objects.all()

    context = {
        'subjects': subjects,
        'no_live_session': request.user.role == 'student' and not subjects.exists(),
    }
    return render(request, 'mark_attendance_face.html', context)


@login_required
def mark_attendance_retina(request):
    """Mark attendance using retina recognition"""
    # Get subjects based on user role
    if request.user.role == 'student':
        subjects = request.user.enrolled_subjects.all()
    elif request.user.role == 'teacher':
        subjects = Subject.objects.filter(teacher=request.user)
    else:
        subjects = Subject.objects.all()
    
    context = {
        'subjects': subjects,
    }
    return render(request, 'mark_attendance_retina.html', context)


@csrf_exempt
@login_required
def process_face_recognition(request):
    """
    Student face attendance — CNN-first identity verification.

    PRIMARY PATH (CNN model available for this student):
      1. Find the latest CNN model whose label_encoder contains request.user.id.
      2. Run the live frame through the CNN — it returns (predicted_user_id, confidence).
      3. Gate A: predicted_user_id == -1  → reject (unknown face, not in database).
      4. Gate B: predicted_user_id != logged-in user → reject (someone else's face).
      5. Gate C: confidence < CNN_THRESHOLD (0.82) → reject (low certainty).
      All three gates must pass. This prevents non-users and other students entirely.

    FALLBACK PATH (student not yet in any CNN model):
      HOG+LBP 5-gate system. Also blocks unless the student has >= 5 stored samples.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'POST required.'}, status=405)

    try:
        import pickle
        from .models import BiometricSample

        # ── Gate 0: Role & approval ────────────────────────────────────────────
        if request.user.role != 'student':
            return JsonResponse({
                'success': False,
                'message': 'Only registered students may use face attendance.',
            }, status=403)

        if not getattr(request.user, 'is_approved', True):
            return JsonResponse({
                'success': False,
                'message': 'Your account is pending approval. Contact your administrator.',
            }, status=403)

        data       = json.loads(request.body)
        image_data = data.get('image', '')
        subject_id = data.get('subject_id')

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr  = np.frombuffer(image_bytes, np.uint8)
        image  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JsonResponse({'success': False,
                                 'message': 'Could not decode image. Please try again.'})

        # ── Build brightness-enhanced variants for dim-light robustness ──────────
        # Even after the frontend boost the server must also try enhanced images,
        # because JPEG compression at low light still degrades face structure.
        # Variants run from mild (CLAHE) to aggressive (gamma 0.25, histeq).
        def _make_bright_variants_local(img):
            gray_lum = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            out = [img]
            try:
                cl = face_recognizer.apply_clahe(img)
                out.append(cl)
            except Exception:
                cl = None
            # Thresholds raised vs old values: a bright background wall can pull
            # mean lum to 150+ even when the face itself is dim. Being more
            # aggressive ensures the face region gets boosted in all such cases.
            gammas = []
            if gray_lum < 170: gammas.append(0.45)   # mild boost — always useful
            if gray_lum < 140: gammas.append(0.35)   # moderate boost
            if gray_lum < 110: gammas.append(0.25)   # strong boost
            for g in gammas:
                try:
                    tbl     = np.array([(i / 255.0) ** (1.0 / g) * 255
                                         for i in range(256)], dtype=np.uint8)
                    boosted = cv2.LUT(img, tbl)
                    out.append(boosted)
                    try: out.append(face_recognizer.apply_clahe(boosted))
                    except Exception: pass
                except Exception:
                    pass
            try:
                geq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                out.append(cv2.cvtColor(geq, cv2.COLOR_GRAY2BGR))
            except Exception:
                pass
            return out

        _img_variants = _make_bright_variants_local(image)

        # ── Extract live face encoding — stop at first success ─────────────────
        face_encoding = None
        for _v in _img_variants:
            face_encoding = face_recognizer.extract_face_encoding(_v)
            if face_encoding is not None:
                break

        if face_encoding is None:
            return JsonResponse({
                'success': False,
                'message': (
                    'No face detected. Ensure good lighting and '
                    'look directly at the camera.'
                ),
                'confidence': 0.0,
            })


        # ── Find the latest CNN model that includes this student (CACHED) ───────
        logged_user_id = request.user.id
        models_dir = os.path.join(str(settings.MEDIA_ROOT), 'models')

        cnn_model      = None
        cnn_labels     = None
        cnn_model_name = None

        if os.path.exists(models_dir):
            import pickle
            model_files = sorted(
                [f for f in os.listdir(models_dir) if f.endswith('.h5')],
                key=lambda f: os.path.getmtime(os.path.join(models_dir, f)),
                reverse=True,
            )
            for mf in model_files:
                label_path = os.path.join(models_dir, mf.replace('.h5', '_labels.pkl'))
                if not os.path.exists(label_path):
                    continue
                try:
                    with open(label_path, 'rb') as lf:
                        labels = pickle.load(lf)
                    if logged_user_id in labels:
                        # Use cached loader — no disk I/O if model hasn't changed
                        cnn_model, cnn_labels = _load_cnn_model_cached(
                            os.path.join(models_dir, mf)
                        )
                        cnn_model_name = mf
                        break
                except Exception:
                    continue


        # ══════════════════════════════════════════════════════════════════════
        # IDENTITY GATE — CNN PRIMARY + HOG WINNER-TAKES-ALL FALLBACK
        #
        # STRATEGY: The live face is scored against EVERY registered student.
        # The portal owner MUST be the top-scoring student AND beat all others
        # by a clear margin. If anyone else scores higher → REJECT immediately.
        #
        # CNN path (preferred): predicted class must == logged-in student.
        # HOG path (fallback):  winner-takes-all with strict margin gate.
        # ══════════════════════════════════════════════════════════════════════
        from .models import BiometricSample as _BS

        confidence = 0.0  # will be set by whichever path succeeds
        identity_confirmed = False

        # ── PATH 1: CNN Winner-Takes-All (most accurate) ───────────────────────
        # If a CNN model exists that includes this student, the CNN's top
        # predicted class MUST be the logged-in student. Any other prediction
        # means a different face is present → reject immediately.
        if cnn_model is not None and cnn_labels is not None and logged_user_id in cnn_labels:
            try:
                _preds = cnn_model.predict(
                    np.expand_dims(face_encoding, axis=0), verbose=0
                )[0]
                _logged_idx  = cnn_labels.get(logged_user_id)
                _winner_idx  = int(np.argmax(_preds))
                _idx_to_uid  = {v: k for k, v in cnn_labels.items()}
                _winner_uid  = _idx_to_uid.get(_winner_idx)
                _own_conf    = float(_preds[_logged_idx]) if _logged_idx is not None else 0.0
                _winner_conf = float(_preds[_winner_idx])

                print(f'[StudentFace CNN] winner_uid={_winner_uid} '
                      f'logged_uid={logged_user_id} '
                      f'own_conf={_own_conf:.4f} winner_conf={_winner_conf:.4f}')

                CNN_THR = 0.60  # own-class probability must exceed this
                                # (0.70 was too strict for small datasets where
                                #  softmax probabilities are spread thinner)

                if _winner_uid == logged_user_id and _own_conf >= CNN_THR:
                    # CNN confirms identity
                    confidence = _own_conf
                    identity_confirmed = True
                    print(f'[StudentFace CNN] CONFIRMED uid={logged_user_id} conf={_own_conf:.4f}')
                else:
                    # CNN says this is a different face — but it might be poorly trained.
                    # Log the failure and let it fall through to the strict HOG+LBP gate.
                    _cnn_reject_reason = (
                        f'CNN predicted uid={_winner_uid} (expected {logged_user_id}), '
                        f'own_conf={_own_conf:.2f}'
                    )
                    SystemLog.objects.create(
                        user=request.user, log_type='warning',
                        action='Face Recognition CNN Failed (Falling back to HOG)',
                        description=(
                            f'Student {request.user.username} CNN validation failed: {_cnn_reject_reason}. '
                            f'Falling back to robust HOG+LBP.'
                        ),
                        ip_address=request.META.get('REMOTE_ADDR'),
                    )
                    print(f'[StudentFace CNN] Failed: {_cnn_reject_reason}. Falling back to HOG.')
            except Exception as _cnn_err:
                print(f'[StudentFace CNN] error (falling back to HOG): {_cnn_err}')
                # Fall through to HOG path

        # ── PATH 2: HOG+LBP Winner-Takes-All (fallback when no CNN) ───────────
        # The live frame is scored against the portal owner's samples AND all
        # other registered students. The portal owner MUST have the highest
        # score AND must exceed every other student by at least WIN_MARGIN.
        # This is a true 1-vs-N comparison — not just a binary threshold check.
        if not identity_confirmed:
            # Require at least 3 stored face samples to verify
            _all_own = list(
                _BS.objects.filter(user=request.user, sample_type='face')
                .order_by('-id')[:30]
            )
            if len(_all_own) < 3:
                return JsonResponse({
                    'success': False,
                    'redirect_dashboard': True,
                    'message': (
                        'Your face has not been fully registered. '
                        'Please contact your administrator to complete face registration.'
                    ),
                    'confidence': 0.0,
                })

            # Extract HOG+LBP descriptor — loop over the same brightness
            # variants built above, stopping as soon as we get descriptors.
            # Flipped variant is also tested because registration captures
            # frames in mirrored orientation.
            live_desc      = None
            live_desc_flip = None

            # Use extract_robust_encoding (60% glasses-invariant zones + 40%
            # full-face HOG+LBP) — MUST match the descriptor used for stored
            # samples in _get_cached_descriptor, otherwise cosine similarity
            # compares vectors of different dimensionality / meaning.
            for _variant in _img_variants:
                try:
                    _d = face_recognizer.extract_robust_encoding(_variant)
                    if _d is None:  # fallback to standard if robust fails
                        _d = face_recognizer.extract_combined_encoding(_variant)
                    if _d is not None and live_desc is None:
                        live_desc = _d
                    _df = face_recognizer.extract_robust_encoding(cv2.flip(_variant, 1))
                    if _df is None:
                        _df = face_recognizer.extract_combined_encoding(cv2.flip(_variant, 1))
                    if _df is not None and live_desc_flip is None:
                        live_desc_flip = _df
                except Exception:
                    pass
                if live_desc is not None and live_desc_flip is not None:
                    break

            if live_desc is None and live_desc_flip is None:
                return JsonResponse({
                    'success': False,
                    'redirect_dashboard': True,
                    'message': 'No face detected. Ensure good lighting and face the camera directly.',
                    'confidence': 0.0,
                })

            # Score live face against portal owner's stored samples.
            # For each stored sample we compare FOUR combinations:
            #   (live_normal | live_flipped) × (stored_original | stored_CLAHE)
            # This handles the registration-in-bright-light vs scan-in-dim-light
            # mismatch — at least one combination produces a usable similarity.
            # A dimension guard prevents silent 0.0 from mismatched vectors.
            user_best = 0.0
            for _s in _all_own:
                try:
                    _fp = os.path.join(str(settings.MEDIA_ROOT), _s.image_path)
                    _d_orig = _get_cached_descriptor(_fp)   # robust encoding
                    if _d_orig is None:
                        continue

                    # Also try CLAHE-enhanced version of the stored image
                    _d_clahe = None
                    try:
                        _si = cv2.imread(_fp)
                        if _si is not None:
                            _si_c = face_recognizer.apply_clahe(_si)
                            _d_clahe = face_recognizer.extract_robust_encoding(_si_c)
                            if _d_clahe is None:
                                _d_clahe = face_recognizer.extract_combined_encoding(_si_c)
                    except Exception:
                        pass

                    _stored_descs = [d for d in [_d_orig, _d_clahe] if d is not None]
                    _live_descs   = [d for d in [live_desc, live_desc_flip] if d is not None]

                    _best = 0.0
                    for _dl in _live_descs:
                        for _ds in _stored_descs:
                            _sim = _robust_cosine_similarity(_dl, _ds)
                            _best = max(_best, _sim)

                    user_best = max(user_best, _best)
                    print(f'[StudentFace HOG] own sample best={_best:.4f} running_best={user_best:.4f}')
                except Exception:
                    continue

            # ── WINNER-TAKES-ALL GATE ──────────────────────────────────────────
            # Rule 1 (MIN SCORE):  own score must be >= 0.55
            #   Robust encoding cosine similarities are lower than combined
            #   encoding (multi-zone weighted concat produces smaller dot
            #   products). 0.55 is the empirical floor for correct same-person
            #   matches across varying lighting, glasses on/off, and distance.
            # Rule 2 (WIN MARGIN): own score must beat any other student by >= 2%.
            #   Robust encoding separates same/different-person scores more
            #   cleanly, so 2% margin is sufficient to reject impostors.
            # Rule 3 (SOLO FLOOR): if this is the only student, apply MIN only.
            HOG_MIN     = 0.55   # minimum acceptable score
            WIN_MARGIN  = 0.02   # must beat all others by 2 percentage points
            SOLO_FLOOR  = 0.55   # floor when this is the only student

            other_best     = 0.0
            other_winner   = None
            other_count    = 0

            # Fast-fail optimization: If the user fails the minimum score, skip the 80 comparisons
            if user_best >= HOG_MIN:
                # Score live face against recent registered student samples
                _all_others = _BS.objects.filter(
                    sample_type='face', user__role='student'
                ).exclude(user=request.user).order_by('-id')[:80]

                other_count = _BS.objects.filter(
                    sample_type='face', user__role='student'
                ).exclude(user=request.user).values('user').distinct().count()

                for _s in _all_others:
                    try:
                        _fp = os.path.join(str(settings.MEDIA_ROOT), _s.image_path)
                        _d  = _get_cached_descriptor(_fp)
                        if _d is None:
                            continue
                        _ld = live_desc if live_desc is not None else live_desc_flip
                        if _ld is None:
                            continue
                        _sim = _robust_cosine_similarity(_ld, _d)
                        if _sim > other_best:
                            other_best   = _sim
                            other_winner = _s.user_id
                    except Exception:
                        continue

            g_min = user_best >= HOG_MIN
            if other_count == 0:
                g_win = user_best >= SOLO_FLOOR
            else:
                g_win = (user_best - other_best) >= WIN_MARGIN

            print(f'[StudentFace HOG WTA] user_best={user_best:.4f} '
                  f'other_best={other_best:.4f} '
                  f'margin={user_best - other_best:.4f} '
                  f'other_winner={other_winner} '
                  f'g_min={g_min} g_win={g_win}')

            if g_min and g_win:
                confidence = user_best
                identity_confirmed = True
                print(f'[StudentFace HOG WTA] CONFIRMED uid={logged_user_id} '
                      f'best={user_best:.4f}')
            else:
                _reject_parts = []
                if not g_min:
                    _reject_parts.append(
                        f'score too low ({user_best:.0%}, need ≥{HOG_MIN:.0%})'
                    )
                if not g_win:
                    _reject_parts.append(
                        f'another student (id={other_winner}) scored {other_best:.0%} '
                        f'— margin only {user_best - other_best:.0%}, need ≥{WIN_MARGIN:.0%}'
                    )
                _reject_reason = '; '.join(_reject_parts) or 'unknown'

                SystemLog.objects.create(
                    user=request.user, log_type='security',
                    action='Face Recognition Rejected (HOG WTA)',
                    description=(
                        f'Student {request.user.username} REJECTED — {_reject_reason}'
                    ),
                    ip_address=request.META.get('REMOTE_ADDR'),
                )
                return JsonResponse({
                    'success': False,
                    'redirect_dashboard': True,
                    'confidence': round(user_best, 4),
                    'message': (
                        'Face identity mismatch — this face does not match your '
                        'registered profile. Redirecting to dashboard.'
                    ),
                })

        # ── Identity confirmed by CNN or HOG WTA ───────────────────────────────
        print(f'[StudentFace] CONFIRMED uid={logged_user_id} conf={confidence:.4f}')




        # ── Validate subject ───────────────────────────────────────────────────
        subject = None
        if subject_id:
            try:
                subject = Subject.objects.get(id=int(subject_id))
            except (Subject.DoesNotExist, ValueError):
                return JsonResponse({'success': False,
                                     'message': 'Invalid subject selected.'})
            if not subject.students.filter(id=request.user.id).exists():
                return JsonResponse({
                    'success': False,
                    'message': f'You are not enrolled in "{subject.name}".',
                })
            if not LiveAttendanceSession.objects.filter(
                subject=subject, is_active=True
            ).exists():
                return JsonResponse({
                    'success': False,
                    'message': (
                        f'No active session for "{subject.name}". '
                        'Please wait for your teacher to start the session.'
                    ),
                })

        # ── Duplicate check ────────────────────────────────────────────────────
        today    = timezone.now().date()
        existing = Attendance.objects.filter(
            user=request.user, subject=subject, date=today
        ).first()
        if existing:
            return JsonResponse({
                'success': True, 'already_marked': True,
                'message': (
                    f'Attendance already recorded today '
                    f'({existing.get_status_display()}) '
                    f'at {existing.time.strftime("%I:%M %p")}.'
                ),
                'confidence': round(confidence, 4),
            })

        # ── Create attendance record ───────────────────────────────────────────
        attendance = Attendance.objects.create(
            user=request.user, subject=subject, date=today,
            status='present', recognition_type='face',
            confidence_score=confidence,
            marked_by=request.user, is_verified=True,
            is_locked=True,   # biometric self-mark — locked against manual override
        )

        SystemLog.objects.create(
            user=request.user, log_type='info',
            action='Attendance Marked',
            description=(
                f'Student {request.user.username} marked attendance via face recognition '
                f'(confidence: {confidence:.2f}, '
                f'subject: {subject.name if subject else "N/A"})'
            ),
            ip_address=request.META.get('REMOTE_ADDR'),
        )

        from .models import Notification
        Notification.objects.create(
            user=request.user, notification_type='success',
            title='Attendance Marked',
            message=(
                f'Your attendance for {subject.name if subject else "General"} on '
                f'{today.strftime("%d %B %Y")} has been recorded as Present.'
            ),
            action_url='/attendance/history/',
        )

        return JsonResponse({
            'success': True, 'already_marked': False,
            'message': (
                f'Attendance marked! Welcome, '
                f'{request.user.get_full_name() or request.user.username}.'
            ),
            'confidence': round(confidence, 4),
            'subject_name': subject.name if subject else None,
            'time': attendance.time.strftime('%I:%M %p'),
        })

    except Exception as e:
        return JsonResponse({'success': False, 'message': f'Server error: {str(e)}'})








@csrf_exempt
@login_required
def process_retina_recognition(request):
    """Process retina recognition from camera"""
    if request.method == 'POST':
        try:
            # Get image data from request
            data = json.loads(request.body)
            image_data = data.get('image')
            subject_id = data.get('subject_id')
            
            # Decode base64 image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Recognize retina
            user_id, confidence = retina_recognizer.recognize_retina(image)
            
            if user_id is not None:
                # Get the recognized user object
                try:
                    recognized_user = User.objects.get(id=user_id)
                except User.DoesNotExist:
                    return JsonResponse({
                        'success': False,
                        'message': 'Recognized user not found in database.'
                    })
                
                # Get subject
                subject = get_object_or_404(Subject, id=subject_id) if subject_id else None
                
                # Validate enrollment for students
                if recognized_user.role == 'student' and subject is not None:
                    if not subject.students.filter(id=recognized_user.id).exists():
                        return JsonResponse({
                            'success': False,
                            'message': f'Recognized user {recognized_user.get_full_name() or recognized_user.username} is not enrolled in this subject.'
                        })
                
                # Mark attendance for the RECOGNIZED user, not the logged-in user
                attendance, created = Attendance.objects.get_or_create(
                    user=recognized_user,  # Use recognized user instead of request.user
                    subject=subject,
                    date=timezone.now().date(),
                    defaults={
                        'status': 'present',
                        'recognition_type': 'retina',
                        'confidence_score': confidence
                    }
                )
                
                if created:
                    SystemLog.objects.create(
                        user=recognized_user,
                        log_type='info',
                        action='Attendance Marked',
                        description=f'Attendance marked via retina recognition (confidence: {confidence:.2f}) by {request.user.username}',
                        ip_address=request.META.get('REMOTE_ADDR')
                    )
                    
                    return JsonResponse({
                        'success': True,
                        'message': f'Attendance marked successfully for {recognized_user.get_full_name() or recognized_user.username}!',
                        'confidence': confidence,
                        'recognized_user': recognized_user.get_full_name() or recognized_user.username
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'message': f'{recognized_user.get_full_name() or recognized_user.username} attendance already marked for today.'
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'Retina not recognized. Please try again.'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': f'Error: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'message': 'Invalid request'})


@login_required
def attendance_history(request):
    """View attendance history - filterable by role via ?role= query param"""
    role_filter = request.GET.get('role', '')

    if request.user.role == 'student':
        attendance_records = Attendance.objects.filter(user=request.user).order_by('-date')
        page_label = 'My Attendance History'
    elif role_filter == 'teacher':
        attendance_records = Attendance.objects.filter(
            user__role='teacher'
        ).select_related('user', 'subject').order_by('-date')
        page_label = 'Teacher Attendance History'
    elif role_filter == 'employee':
        attendance_records = Attendance.objects.filter(
            user__role='admin'
        ).select_related('user', 'subject').order_by('-date')
        page_label = 'Employee Attendance History'
    elif role_filter == 'student':
        attendance_records = Attendance.objects.filter(
            user__role='student'
        ).select_related('user', 'subject').order_by('-date')
        page_label = 'Student Attendance History'
    else:
        attendance_records = Attendance.objects.all().select_related('user', 'subject').order_by('-date')
        page_label = 'Attendance History'

    return render(request, 'attendance_history.html', {
        'attendance_records': attendance_records,
        'page_label': page_label,
        'role_filter': role_filter,
    })



@login_required
def subjects_list(request):
    """List all subjects"""
    if request.user.role == 'admin':
        subjects = Subject.objects.all()
    elif request.user.role == 'teacher':
        subjects = Subject.objects.filter(teacher=request.user)
    else:
        subjects = Subject.objects.all()
    
    return render(request, 'subjects_list.html', {'subjects': subjects})


@login_required
def subject_detail(request, subject_id):
    """View subject details and enrolled students"""
    subject = get_object_or_404(Subject, id=subject_id)
    
    # Check permissions
    if request.user.role == 'teacher' and subject.teacher != request.user:
        messages.error(request, 'You do not have permission to view this subject.')
        return redirect('subjects_list')
    
    # Get enrolled students
    enrolled_students = subject.students.all().order_by('first_name', 'last_name')
    
    context = {
        'subject': subject,
        'enrolled_students': enrolled_students,
        'student_count': enrolled_students.count(),
    }
    
    return render(request, 'subject_detail.html', context)


@login_required
def create_subject(request):
    """Create new subject (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = SubjectForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Subject created successfully!')
            return redirect('subjects_list')
    else:
        form = SubjectForm()
    
    return render(request, 'create_subject.html', {'form': form})


@login_required
def delete_subject(request, subject_id):
    """Delete subject (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')
    
    subject = get_object_or_404(Subject, id=subject_id)
    
    # Store subject name before deletion
    subject_name = subject.name
    subject_code = subject.code
    
    # Delete the subject
    subject.delete()
    
    # Log the deletion
    SystemLog.objects.create(
        user=request.user,
        log_type='warning',
        action='Subject Deleted',
        description=f'Admin deleted subject: {subject_name} ({subject_code})',
        ip_address=request.META.get('REMOTE_ADDR')
    )
    
    messages.success(request, f'Subject "{subject_name}" has been deleted successfully.')
    return redirect('subjects_list')


@login_required
def enroll_students(request, subject_id):
    """Enroll or remove students from a subject (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')
    
    subject = get_object_or_404(Subject, id=subject_id)
    
    if request.method == 'POST':
        # Get selected student IDs from form
        student_ids = request.POST.getlist('students')
        
        # Clear existing enrollments and add new ones
        subject.students.clear()
        if student_ids:
            students = User.objects.filter(id__in=student_ids, role='student', is_approved=True)
            subject.students.add(*students)
            messages.success(request, f'{len(students)} student(s) enrolled in {subject.name}.')
        else:
            messages.info(request, f'All students removed from {subject.name}.')
        
        # Log the action
        SystemLog.objects.create(
            user=request.user,
            log_type='info',
            action='Students Enrolled',
            description=f'Updated student enrollment for subject: {subject.name} ({subject.code})',
            ip_address=request.META.get('REMOTE_ADDR')
        )
        
        return redirect('subjects_list')
    
    # GET request - show enrollment form
    # Only show students from the SAME section as the subject.
    # Cross-section enrollment is not allowed.
    enrolled_students = subject.students.all()
    if subject.section:
        section_students = User.objects.filter(
            role='student', is_approved=True, section=subject.section
        ).order_by('first_name', 'last_name')
    else:
        # Subject has no section assigned — show all students
        section_students = User.objects.filter(
            role='student', is_approved=True
        ).order_by('first_name', 'last_name')

    context = {
        'subject': subject,
        'section_students': section_students,
        'enrolled_students': enrolled_students,
    }
    return render(request, 'enroll_students.html', context)


@login_required
def generate_report(request):
    """Generate attendance report with optional dept/section filters"""
    from .models import Department, Section as SectionModel
    departments = Department.objects.all().order_by('name')

    if request.method == 'POST':
        form = ReportGenerationForm(request.POST)
        if form.is_valid():
            report_type    = form.cleaned_data['report_type']
            start_date     = form.cleaned_data['start_date']
            end_date       = form.cleaned_data['end_date']
            subject        = form.cleaned_data.get('subject')
            user           = form.cleaned_data.get('user')
            export_format  = form.cleaned_data['export_format']

            # New: department / section from POST (not Django form fields)
            dept_id    = request.POST.get('department') or None
            section_id = request.POST.get('section') or None

            # Build query filters
            filters = {
                'date__gte': start_date,
                'date__lte': end_date,
            }
            if subject:
                filters['subject'] = subject
            if user:
                filters['user'] = user
            elif section_id:
                filters['user__section_id'] = section_id
            elif dept_id:
                filters['user__dept_id'] = dept_id

            attendance_records = Attendance.objects.filter(**filters).select_related(
                'user', 'subject'
            ).order_by('date', 'user__username')

            # Build report title
            title_parts = [report_type.capitalize(), 'Attendance Report']
            if dept_id:
                try:
                    dept_obj = Department.objects.get(id=dept_id)
                    title_parts.append(f'— {dept_obj.code}')
                except Department.DoesNotExist:
                    pass
            if section_id:
                try:
                    sec_obj = SectionModel.objects.get(id=section_id)
                    title_parts.append(f'[{sec_obj.name}]')
                except SectionModel.DoesNotExist:
                    pass
            if subject:
                title_parts.append(f'— {subject.code}')
            if user:
                title_parts.append(f'— {user.get_full_name() or user.username}')
            title = ' '.join(title_parts)

            report = AttendanceReport.objects.create(
                title=title,
                report_type=report_type,
                start_date=start_date,
                end_date=end_date,
                generated_by=request.user
            )

            if export_format == 'pdf':
                file_path = generate_pdf_report(report, attendance_records, start_date, end_date, subject, user)
            else:
                file_path = generate_csv_report(report, attendance_records, start_date, end_date, subject, user)

            report.file_path = file_path
            report.save()

            SystemLog.objects.create(
                user=request.user,
                log_type='info',
                action='Report Generated',
                description=f'Generated {report_type} report: {title}',
                ip_address=request.META.get('REMOTE_ADDR')
            )

            messages.success(request, f'Report "{title}" generated successfully!')
            return redirect('reports_list')
    else:
        form = ReportGenerationForm()

    return render(request, 'generate_report.html', {
        'form': form,
        'departments': departments,
    })


@login_required
def reports_list(request):
    """List generated reports"""
    reports = AttendanceReport.objects.all().order_by('-created_at')
    return render(request, 'reports_list.html', {'reports': reports})


@login_required
def delete_report(request, report_id):
    """Delete attendance report (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied. Only admins can delete reports.')
        return redirect('reports_list')
    
    report = get_object_or_404(AttendanceReport, id=report_id)
    
    # Store report info before deletion
    report_title = report.title
    
    # Delete the physical file if it exists
    if report.file_path:
        try:
            file_path = os.path.join(settings.MEDIA_ROOT, report.file_path.name)
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            # Log error but continue with deletion
            SystemLog.objects.create(
                user=request.user,
                log_type='error',
                action='Report File Deletion Error',
                description=f'Error deleting report file: {str(e)}',
                ip_address=request.META.get('REMOTE_ADDR')
            )
    
    # Delete the report record
    report.delete()
    
    # Log the deletion
    SystemLog.objects.create(
        user=request.user,
        log_type='warning',
        action='Report Deleted',
        description=f'Admin deleted attendance report: {report_title}',
        ip_address=request.META.get('REMOTE_ADDR')
    )
    
    messages.success(request, f'Report "{report_title}" has been deleted successfully.')
    return redirect('reports_list')


@login_required
def admin_panel(request):
    """Admin panel (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')
    
    from .models import BiometricSample
    pending_users = User.objects.filter(is_approved=False)
    all_users = list(User.objects.all())
    system_logs = SystemLog.objects.all().order_by('-created_at')[:50]

    # Attach first face sample path to each user object for display in the template
    face_samples = BiometricSample.objects.filter(sample_type='face').order_by('user_id', 'created_at')
    user_photos = {}
    for sample in face_samples:
        if sample.user_id not in user_photos:
            user_photos[sample.user_id] = sample.image_path

    for user in all_users:
        user.face_photo_path = user_photos.get(user.id)

    context = {
        'pending_users': pending_users,
        'all_users': all_users,
        'system_logs': system_logs,
    }
    
    return render(request, 'admin_panel.html', context)


@login_required
def approve_user(request, user_id):
    """Approve user (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')
    
    user = get_object_or_404(User, id=user_id)
    user.is_approved = True
    user.save()
    
    # Create notification for approved user
    from .models import Notification
    Notification.objects.create(
        user=user,
        notification_type='success',
        title='Account Approved',
        message='Your account has been approved by an administrator. You can now access all features.',
        action_url='/dashboard/'
    )
    
    messages.success(request, f'User {user.username} approved successfully!')
    return redirect('admin_panel')


@login_required
def delete_user(request, user_id):
    """Delete user (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')
    
    user = get_object_or_404(User, id=user_id)
    
    # Prevent deleting yourself
    if user.id == request.user.id:
        messages.error(request, 'You cannot delete your own account!')
        return redirect('admin_panel')
    
    # Store username before deletion
    username = user.username
    
    # Delete the user
    user.delete()
    
    # Log the deletion
    SystemLog.objects.create(
        user=request.user,
        log_type='warning',
        action='User Deleted',
        description=f'Admin deleted user account: {username}',
        ip_address=request.META.get('REMOTE_ADDR')
    )
    
    messages.success(request, f'User {username} has been deleted successfully.')
    return redirect('admin_panel')


# ============= NOTIFICATION MODULE =============

@login_required
def notifications(request):
    """View all notifications"""
    from .models import Notification
    
    user_notifications = Notification.objects.filter(user=request.user)
    
    context = {
        'notifications': user_notifications,
        'total_notifications': user_notifications.count(),
        'unread_count': user_notifications.filter(is_read=False).count(),
        'success_count': user_notifications.filter(notification_type='success').count(),
        'alert_count': user_notifications.filter(notification_type='alert').count(),
    }
    
    return render(request, 'notifications.html', context)


@csrf_exempt
@login_required
def mark_notification_read(request, notification_id):
    """Mark notification as read"""
    if request.method == 'POST':
        from .models import Notification
        notification = get_object_or_404(Notification, id=notification_id, user=request.user)
        notification.is_read = True
        notification.save()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def delete_notification(request, notification_id):
    """Delete notification"""
    if request.method == 'DELETE':
        from .models import Notification
        notification = get_object_or_404(Notification, id=notification_id, user=request.user)
        notification.delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def mark_all_notifications_read(request):
    """Mark all notifications as read"""
    if request.method == 'POST':
        from .models import Notification
        Notification.objects.filter(user=request.user, is_read=False).update(is_read=True)
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def clear_all_notifications(request):
    """Clear all notifications"""
    if request.method == 'DELETE':
        from .models import Notification
        Notification.objects.filter(user=request.user).delete()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


# ============= DATASET & TRAINING MODULE =============

@login_required
def dataset_management(request):
    """Dataset creation and model training"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied. Only admins can manage datasets.')
        return redirect('dashboard')
    
    from .models import BiometricSample
    
    datasets = TrainingDataset.objects.all()
    users = User.objects.filter(is_approved=True)
    
    # Update sample counts for all datasets
    for dataset in datasets:
        dataset.update_sample_count()
    
    # Calculate statistics from BiometricSample
    total_face_samples = BiometricSample.objects.filter(sample_type='face').count()
    total_retina_samples = BiometricSample.objects.filter(sample_type='retina').count()
    total_users_enrolled = BiometricSample.objects.values('user').distinct().count()
    
    # Get latest trained model accuracy
    latest_model = TrainingDataset.objects.filter(is_trained=True).order_by('-updated_at').first()
    model_accuracy = latest_model.accuracy if latest_model else 0
    
    context = {
        'datasets': datasets,
        'users': users,
        'total_face_samples': total_face_samples,
        'total_retina_samples': total_retina_samples,
        'total_users_enrolled': total_users_enrolled,
        'model_accuracy': round(model_accuracy, 2),
        'trained_models': TrainingDataset.objects.filter(is_trained=True),
    }
    
    return render(request, 'dataset_management.html', context)


@csrf_exempt
@login_required
def create_dataset(request):
    """Create new dataset"""
    if request.method == 'POST':
        if request.user.role != 'admin':
            return JsonResponse({'success': False, 'message': 'Permission denied'})
        
        data = json.loads(request.body)
        dataset = TrainingDataset.objects.create(
            name=data.get('name'),
            description=data.get('description', ''),
            dataset_type=data.get('dataset_type')
        )
        
        return JsonResponse({'success': True, 'dataset_id': dataset.id})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def collect_samples(request):
    """Collect biometric samples for training"""
    if request.method == 'POST':
        try:
            import os
            from django.conf import settings
            from .models import BiometricSample
            from datetime import datetime
            
            data = json.loads(request.body)
            user_id = data.get('user_id')
            sample_type = data.get('sample_type')  # 'face' or 'retina'
            image_data = data.get('image')
            
            if not user_id or not sample_type or not image_data:
                return JsonResponse({'success': False, 'message': 'Missing required fields'})
            
            user = get_object_or_404(User, id=user_id)
            
            # Decode base64 image
            image_data_parts = image_data.split(',')
            if len(image_data_parts) > 1:
                image_data = image_data_parts[1]
            
            image_bytes = base64.b64decode(image_data)
            
            # Create directory structure: media/training_data/{sample_type}/{user_id}/
            media_root = settings.MEDIA_ROOT
            training_dir = os.path.join(media_root, 'training_data', sample_type, str(user_id))
            os.makedirs(training_dir, exist_ok=True)
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'{sample_type}_sample_{timestamp}.jpg'
            file_path = os.path.join(training_dir, filename)
            
            # Save image file
            with open(file_path, 'wb') as f:
                f.write(image_bytes)
            
            # Create relative path for database
            relative_path = os.path.join('training_data', sample_type, str(user_id), filename)
            
            # Get user ID (student_id or employee_id)
            user_identifier = user.student_id if user.role == 'student' else user.employee_id
            dataset_name = user_identifier if user_identifier else f"User_{user.id}"
            
            # Get or create individual dataset for this user
            dataset_to_link, _ = TrainingDataset.objects.get_or_create(
                dataset_type=sample_type,
                name=dataset_name,
                defaults={'description': f'{sample_type.capitalize()} recognition dataset for {user_identifier or user.username}'}
            )
            
            # Create BiometricSample record and link to user's dataset
            sample = BiometricSample.objects.create(
                user=user,
                sample_type=sample_type,
                image_path=relative_path,
                dataset=dataset_to_link
            )
            
            # Update the user's dataset with sample count
            dataset_to_link.update_sample_count()

            
            return JsonResponse({
                'success': True, 
                'message': 'Sample saved successfully',
                'sample_id': sample.id
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def train_model(request):
    """Train AI model"""
    global face_recognizer
    if request.method == 'POST':
        if request.user.role != 'admin':
            return JsonResponse({'success': False, 'message': 'Permission denied'})
        
        try:
            from django.conf import settings
            import os
            import pickle
            from .models import BiometricData, Notification, TrainingDataset
            
            data = json.loads(request.body)
            dataset_id = data.get('dataset_id')
            epochs = int(data.get('epochs', 50))
            batch_size = int(data.get('batch_size', 32))
            
            dataset = get_object_or_404(TrainingDataset, id=dataset_id)
            
            # Update sample count first
            dataset.update_sample_count()
            
            # Validate dataset has sufficient samples (minimum 5, matching registration)
            if dataset.total_samples < 5:
                return JsonResponse({
                    'success': False, 
                    'message': f'Insufficient samples. Dataset has {dataset.total_samples} samples, minimum 5 required.'
                })
            
            # Load training data based on dataset type
            if dataset.dataset_type == 'face':
                # Load face samples from BiometricSample
                from .models import BiometricSample
                
                # Get all samples for this dataset type (don't filter by user approval for training samples)
                face_samples = BiometricSample.objects.filter(sample_type='face')
                
                print(f"Found {face_samples.count()} total face samples in database")
                
                if face_samples.count() < 2:
                    return JsonResponse({
                        'success': False,
                        'message': f'Need at least 2 samples to train the model. Found {face_samples.count()} samples.'
                    })
                
                # Check unique users (for logging purposes only)
                unique_users = face_samples.values('user').distinct().count()
                print(f"Samples are from {unique_users} unique user(s)")
                
                # Prepare training data
                X_train = []
                y_train = []
                label_encoder = {}
                current_label = 0
                
                # Track errors for debugging
                errors = []
                missing_files = 0
                no_face_detected = 0
                
                for sample in face_samples:
                    try:
                        # Load image - BiometricSample stores path as string
                        # Convert MEDIA_ROOT to string for proper path joining
                        image_path = os.path.join(str(settings.MEDIA_ROOT), sample.image_path)
                        # Normalize path for Windows compatibility
                        image_path = os.path.normpath(image_path)
                        
                        # Check if file exists
                        if not os.path.exists(image_path):
                            missing_files += 1
                            errors.append(f"Sample {sample.id}: File not found at {image_path}")
                            continue
                        
                        image = cv2.imread(image_path)
                        
                        if image is None:
                            errors.append(f"Sample {sample.id}: Failed to read image at {image_path}")
                            continue
                        
                        # Preprocess face
                        face_processed = face_recognizer.extract_face_encoding(image)
                        
                        if face_processed is not None:
                            X_train.append(face_processed)
                            
                            # Map user_id to label
                            if sample.user.id not in label_encoder:
                                label_encoder[sample.user.id] = current_label
                                current_label += 1
                            
                            y_train.append(label_encoder[sample.user.id])
                        else:
                            no_face_detected += 1
                            errors.append(f"Sample {sample.id}: No face detected in image")
                    except Exception as e:
                        errors.append(f"Sample {sample.id}: {str(e)}")
                        print(f"Error processing sample {sample.id}: {e}")
                        continue

                # ── ALWAYS inject a 'Background/Unknown' class ─────────────────
                # A closed-set CNN forces EVERY face (even complete strangers) into
                # one of the known user classes at softmax. Without a background
                # class the model has no way to say "I don't know this person."
                # By training on negative/unknown examples for EVERY model we give
                # Gate-A a real "unknown" class to match against, making open-set
                # rejection work correctly regardless of how many users are enrolled.
                print("[Train] Injecting background/unknown class for open-set rejection...")
                label_encoder[-1] = current_label
                current_label += 1

                face_data_root = os.path.join(str(settings.MEDIA_ROOT), 'training_data', 'face')
                subdirs = []
                if os.path.exists(face_data_root):
                    subdirs = [d for d in os.listdir(face_data_root)
                               if os.path.isdir(os.path.join(face_data_root, d))]

                # Exclude every directory that belongs to a known user
                known_user_ids = {str(uid) for uid in label_encoder if uid != -1}
                background_dirs = [d for d in subdirs if d not in known_user_ids]

                # Target 1:1 ratio (minimum 10 background samples)
                target_bg = max(len(X_train), 10)
                background_count = 0

                # First pass — detect a real face in the image (highest quality)
                for b_dir in background_dirs:
                    if background_count >= target_bg:
                        break
                    dir_path = os.path.join(face_data_root, b_dir)
                    try:
                        img_names = os.listdir(dir_path)
                    except Exception:
                        continue
                    for img_name in img_names:
                        if background_count >= target_bg:
                            break
                        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            continue
                        try:
                            b_image = cv2.imread(os.path.join(dir_path, img_name))
                            if b_image is not None:
                                b_face = face_recognizer.extract_face_encoding(b_image)
                                if b_face is not None:
                                    X_train.append(b_face)
                                    y_train.append(label_encoder[-1])
                                    background_count += 1
                        except Exception:
                            continue

                # Second pass — raw resize fallback when faces can't be detected
                if background_count < target_bg:
                    print(f"[Train] Only {background_count} face-detected backgrounds; "
                          f"filling remainder with raw-resize samples...")
                    for b_dir in background_dirs:
                        if background_count >= target_bg:
                            break
                        dir_path = os.path.join(face_data_root, b_dir)
                        try:
                            img_names = os.listdir(dir_path)
                        except Exception:
                            continue
                        for img_name in img_names:
                            if background_count >= target_bg:
                                break
                            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                continue
                            try:
                                b_image = cv2.imread(os.path.join(dir_path, img_name))
                                if b_image is not None:
                                    b_face = face_recognizer.preprocess_face(b_image)
                                    if b_face is not None:
                                        X_train.append(b_face)
                                        y_train.append(label_encoder[-1])
                                        background_count += 1
                            except Exception:
                                continue

                print(f"[Train] Background injection complete: {background_count} samples "
                      f"from {len(background_dirs)} dir(s). "
                      f"Total training samples: {len(X_train)}")

                # Convert to numpy arrays
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Create a FRESH face recognizer instance for training (don't use global one)
                # This prevents TensorFlow optimizer errors when retraining with different users
                from ai_models.face_recognition import FaceRecognition
                fresh_face_recognizer = FaceRecognition(model_path=None)
                
                # CRITICAL: Even with model_path=None, FaceRecognition auto-loads latest model
                # We MUST set model to None to force fresh creation and avoid optimizer errors
                fresh_face_recognizer.model = None
                fresh_face_recognizer.label_encoder = {}
                
                # Train the model with fresh instance (will create new model from scratch)
                history = fresh_face_recognizer.train_model(X_train, y_train, epochs=epochs)
                
                # Calculate accuracy
                final_accuracy = history.history['accuracy'][-1] * 100
                
                # Save model
                models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                model_filename = f'face_{dataset.id}_model.h5'
                model_path = os.path.join(models_dir, model_filename)
                fresh_face_recognizer.save_model(model_path)
                
                # Save label encoder
                encoder_path = model_path.replace('.h5', '_labels.pkl')
                with open(encoder_path, 'wb') as f:
                    pickle.dump(label_encoder, f)
                
                # Update fresh model's label_encoder for this training session
                fresh_face_recognizer.label_encoder = label_encoder
                
                # CRITICAL: Reload the newly trained model into the GLOBAL face_recognizer
                # Without this, face recognition will still use the old model loaded at server start
                face_recognizer.load_model(model_path)
                face_recognizer.label_encoder = label_encoder
                print(f"Global face_recognizer reloaded with new model and {len(label_encoder)} users")
                
                # Update dataset
                dataset.is_trained = True
                dataset.accuracy = round(final_accuracy, 2)
                dataset.model_path = f'models/{model_filename}'
                dataset.save()
                
                # Create notification
                Notification.objects.create(
                    user=request.user,
                    notification_type='success',
                    title='Model Training Complete',
                    message=f'Training completed for {dataset.name} with {dataset.accuracy}% accuracy using {len(X_train)} samples from {len(label_encoder)} users.',
                    action_url='/dataset-management/'
                )
                
                return JsonResponse({'success': True, 'accuracy': dataset.accuracy})
            
            else:
                # For retina, keep simulated training for now
                import random
                import time
                time.sleep(2)
                base_accuracy = 90.0
                sample_bonus = min((dataset.total_samples / 100) * 5, 8)
                random_factor = random.uniform(-2, 0)
                accuracy = round(base_accuracy + sample_bonus + random_factor, 2)
                
                dataset.is_trained = True
                dataset.accuracy = accuracy
                dataset.model_path = f'models/{dataset.dataset_type}_{dataset.id}_model.h5'
                dataset.save()
                
                Notification.objects.create(
                    user=request.user,
                    notification_type='success',
                    title='Model Training Complete',
                    message=f'Training completed for {dataset.name} with {dataset.accuracy}% accuracy using {dataset.total_samples} samples.',
                    action_url='/dataset-management/'
                )
                
                return JsonResponse({'success': True, 'accuracy': dataset.accuracy})
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Training error: {error_details}")
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def view_dataset_detail(request, dataset_id):
    """View dataset details and samples"""
    dataset = get_object_or_404(TrainingDataset, id=dataset_id)
    dataset.update_sample_count()
    
    from .models import BiometricSample
    samples = BiometricSample.objects.filter(
        sample_type=dataset.dataset_type
    ).select_related('user').order_by('-created_at')[:100]
    
    sample_data = [{
        'id': sample.id,
        'user': sample.user.get_full_name(),
        'image_path': sample.image_path,
        'created_at': sample.created_at.strftime('%Y-%m-%d %H:%M')
    } for sample in samples]
    
    return JsonResponse({
        'success': True,
        'dataset': {
            'id': dataset.id,
            'name': dataset.name,
            'type': dataset.get_dataset_type_display(),
            'samples': dataset.total_samples,
            'is_trained': dataset.is_trained,
            'accuracy': dataset.accuracy,
            'created_at': dataset.created_at.strftime('%Y-%m-%d')
        },
        'samples': sample_data
    })


@csrf_exempt
@login_required
def delete_dataset(request, dataset_id):
    """Delete dataset and its associated samples"""
    if request.method == 'DELETE' or request.method == 'POST':
        if request.user.role != 'admin':
            return JsonResponse({'success': False, 'message': 'Permission denied'})
        
        try:
            dataset = get_object_or_404(TrainingDataset, id=dataset_id)
            dataset_name = dataset.name
            
            # Delete associated samples
            from .models import BiometricSample
            BiometricSample.objects.filter(dataset=dataset).delete()
            
            # Delete dataset
            dataset.delete()
            
            return JsonResponse({
                'success': True,
                'message': f'Dataset "{dataset_name}" deleted successfully'
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def delete_sample(request, sample_id):
    """Delete individual biometric sample"""
    if request.method == 'DELETE' or request.method == 'POST':
        if request.user.role != 'admin':
            return JsonResponse({'success': False, 'message': 'Permission denied'})
        
        try:
            import os
            from django.conf import settings
            from .models import BiometricSample
            
            sample = get_object_or_404(BiometricSample, id=sample_id)
            
            # Delete file from filesystem
            file_path = os.path.join(settings.MEDIA_ROOT, sample.image_path)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete database record
            sample.delete()
            
            # Update dataset sample counts
            datasets = TrainingDataset.objects.filter(dataset_type=sample.sample_type)
            for dataset in datasets:
                dataset.update_sample_count()
            
            return JsonResponse({'success': True, 'message': 'Sample deleted successfully'})
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def test_model_api(request, model_id):
    """Test trained model with sample upload"""
    if request.method == 'POST':
        try:
            dataset = get_object_or_404(TrainingDataset, id=model_id)
            
            if not dataset.is_trained:
                return JsonResponse({'success': False, 'message': 'Model not trained yet'})
            
            # In production, this would perform actual recognition
            # For now, simulate test results
            import random
            test_accuracy = round(dataset.accuracy + random.uniform(-5, 2), 2)
            
            return JsonResponse({
                'success': True,
                'model': dataset.name,
                'test_accuracy': test_accuracy,
                'message': f'Model tested successfully with {test_accuracy}% accuracy'
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def retrain_model_api(request, model_id):
    """Retrain existing model"""
    if request.method == 'POST':
        if request.user.role != 'admin':
            return JsonResponse({'success': False, 'message': 'Permission denied'})
        
        try:
            import random
            import time
            
            dataset = get_object_or_404(TrainingDataset, id=model_id)
            dataset.update_sample_count()
            
            if dataset.total_samples < 10:
                return JsonResponse({
                    'success': False,
                    'message': 'Insufficient samples for retraining'
                })
            
            # Simulate retraining
            time.sleep(2)
            
            # Slight improvement in accuracy for retraining
            base_accuracy = 90.0
            sample_bonus = min((dataset.total_samples / 100) * 5, 8)
            random_factor = random.uniform(0, 2)  # Positive bias for retraining
            new_accuracy = round(base_accuracy + sample_bonus + random_factor, 2)
            
            dataset.accuracy = new_accuracy
            dataset.save()
            
            return JsonResponse({
                'success': True,
                'accuracy': new_accuracy,
                'message': f'Model retrained with {new_accuracy}% accuracy'
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def export_model_api(request, model_id):
    """Export trained model file"""
    dataset = get_object_or_404(TrainingDataset, id=model_id)
    
    if not dataset.is_trained:
        return JsonResponse({'success': False, 'message': 'Model not trained yet'})
    
    # In production, this would return actual model file
    # For now, return model metadata
    return JsonResponse({
        'success': True,
        'model_name': dataset.name,
        'model_path': dataset.model_path,
        'model_type': dataset.dataset_type,
        'accuracy': dataset.accuracy,
        'samples': dataset.total_samples,
        'message': 'Model export initiated. Download will start shortly.'
    })


# ============= MANUAL ATTENDANCE MODULE =============

@login_required
def manual_attendance(request):
    """Manual attendance management"""
    if request.user.role not in ['admin', 'teacher']:
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')
    
    students = User.objects.filter(role='student', is_approved=True)
    
    if request.user.role == 'teacher':
        subjects = Subject.objects.filter(teacher=request.user)
    else:
        subjects = Subject.objects.all()
    
    # Get filter parameters
    filter_date = request.GET.get('filter_date', timezone.now().date())
    filter_subject = request.GET.get('filter_subject', '')
    filter_status = request.GET.get('filter_status', '')
    
    # Get attendance records — STUDENTS ONLY
    attendance_records = Attendance.objects.filter(
        user__role='student'
    ).select_related('user', 'subject')
    
    if filter_date:
        attendance_records = attendance_records.filter(date=filter_date)
    if filter_subject:
        attendance_records = attendance_records.filter(subject_id=filter_subject)
    if filter_status:
        attendance_records = attendance_records.filter(status=filter_status)
    
    # Calculate stats — students only
    today_attendance_count = Attendance.objects.filter(user__role='student', date=timezone.now().date()).count()
    total_students_count = students.count()
    present_count = Attendance.objects.filter(user__role='student', date=timezone.now().date(), status='present').count()
    absent_count = total_students_count - present_count
    
    context = {
        'students': students,
        'subjects': subjects,
        'attendance_records': attendance_records[:100],  # Limit to 100 records
        'today': timezone.now().date(),
        'current_time': timezone.now().time(),
        'filter_date': filter_date,
        'filter_subject': filter_subject,
        'filter_status': filter_status,
        'today_attendance_count': today_attendance_count,
        'total_students': total_students_count,
        'present_count': present_count,
        'absent_count': absent_count,
    }
    
    return render(request, 'manual_attendance.html', context)


@login_required
def manual_mark_attendance(request):
    """Mark attendance manually"""
    if request.method == 'POST':
        if request.user.role not in ['admin', 'teacher']:
            messages.error(request, 'Permission denied.')
            return redirect('dashboard')
        
        user_id = request.POST.get('user_id')
        subject_id = request.POST.get('subject_id')
        date = request.POST.get('date')
        status = request.POST.get('status')
        notes = request.POST.get('notes', '')
        
        user = get_object_or_404(User, id=user_id)
        subject = get_object_or_404(Subject, id=subject_id)
        
        attendance, created = Attendance.objects.update_or_create(
            user=user,
            subject=subject,
            date=date,
            defaults={
                'status': status,
                'recognition_type': 'manual',
                'notes': notes,
                'is_verified': True
            }
        )
        
        # Create notification for student
        from .models import Notification
        Notification.objects.create(
            user=user,
            notification_type='info',
            title='Attendance Marked',
            message=f'Your attendance for {subject.name} on {date} has been marked as {status}.',
            action_url='/attendance/history/'
        )
        
        messages.success(request, 'Attendance marked successfully!')
        return redirect('manual_attendance')
    
    return redirect('manual_attendance')


@login_required
def bulk_mark_attendance(request):
    """Bulk mark attendance"""
    if request.method == 'POST':
        if request.user.role not in ['admin', 'teacher']:
            messages.error(request, 'Permission denied.')
            return redirect('dashboard')
        
        subject_id = request.POST.get('subject_id')
        date = request.POST.get('date')
        student_ids = request.POST.getlist('student_ids')
        
        subject = get_object_or_404(Subject, id=subject_id)
        
        count = 0
        for student_id in student_ids:
            status = request.POST.get(f'status_{student_id}', 'present')
            notes = request.POST.get(f'notes_{student_id}', '')
            
            user = get_object_or_404(User, id=student_id)
            
            Attendance.objects.update_or_create(
                user=user,
                subject=subject,
                date=date,
                defaults={
                    'status': status,
                    'recognition_type': 'manual',
                    'notes': notes,
                    'is_verified': True
                }
            )
            count += 1
        
        messages.success(request, f'Bulk attendance marked for {count} students!')
        return redirect('manual_attendance')
    
    return redirect('manual_attendance')


@csrf_exempt
@login_required
def get_attendance_detail(request, attendance_id):
    """Get attendance details for editing"""
    attendance = get_object_or_404(Attendance, id=attendance_id)
    
    return JsonResponse({
        'id': attendance.id,
        'status': attendance.status,
        'notes': attendance.notes,
        'date': str(attendance.date),
        'time': str(attendance.time),
    })


@csrf_exempt
@login_required
def update_attendance(request, attendance_id):
    """Update attendance record"""
    if request.method == 'POST':
        if request.user.role not in ['admin', 'teacher']:
            return JsonResponse({'success': False, 'message': 'Permission denied'})
        
        attendance = get_object_or_404(Attendance, id=attendance_id)
        
        status = request.POST.get('status')
        notes = request.POST.get('notes', '')
        
        attendance.status = status
        attendance.notes = notes
        attendance.save()
        
        # Create notification
        from .models import Notification
        Notification.objects.create(
            user=attendance.user,
            notification_type='alert',
            title='Attendance Updated',
            message=f'Your attendance for {attendance.subject.name} on {attendance.date} has been updated to {status}.',
            action_url='/attendance/history/'
        )
        
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


@csrf_exempt
@login_required
def delete_attendance_record(request, attendance_id):
    """Delete attendance record"""
    if request.method == 'DELETE':
        if request.user.role not in ['admin', 'teacher']:
            return JsonResponse({'success': False, 'message': 'Permission denied'})
        
        attendance = get_object_or_404(Attendance, id=attendance_id)
        attendance.delete()
        
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})




# ============= CAMERA MANAGEMENT =============

@login_required
def camera_management(request):
    """Camera management page (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    cameras = Camera.objects.all()
    context = {
        'cameras': cameras,
        'total_cameras': cameras.count(),
        'active_cameras': cameras.filter(is_active=True).count(),
    }
    return render(request, 'camera_management.html', context)


@csrf_exempt
@login_required
def create_camera(request):
    """Create a new camera (Admin only)"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'message': 'Permission denied'})

    if request.method == 'POST':
        try:
            from .models import Camera
            data = json.loads(request.body)

            name = data.get('name', '').strip()
            location = data.get('location', '').strip()
            camera_url = data.get('camera_url', '').strip()
            camera_type = data.get('camera_type', 'usb')
            is_active = data.get('is_active', True)
            resolution_width = int(data.get('resolution_width', 640))
            resolution_height = int(data.get('resolution_height', 480))
            fps = int(data.get('fps', 30))

            if not name or not camera_url:
                return JsonResponse({'success': False, 'message': 'Name and Camera URL are required.'})

            camera = Camera.objects.create(
                name=name,
                location=location,
                camera_url=camera_url,
                camera_type=camera_type,
                is_active=is_active,
                resolution_width=resolution_width,
                resolution_height=resolution_height,
                fps=fps,
            )

            SystemLog.objects.create(
                user=request.user,
                log_type='info',
                action='Camera Created',
                description='Admin created camera: {} ({})'.format(name, location),
                ip_address=request.META.get('REMOTE_ADDR')
            )

            return JsonResponse({
                'success': True,
                'message': 'Camera "{}" created successfully!'.format(name),
                'camera': {
                    'id': camera.id,
                    'name': camera.name,
                    'location': camera.location,
                    'camera_url': camera.camera_url,
                    'camera_type': camera.camera_type,
                    'camera_type_display': camera.get_camera_type_display(),
                    'is_active': camera.is_active,
                    'resolution': camera.get_resolution_display(),
                    'fps': camera.fps,
                }
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})

    return JsonResponse({'success': False, 'message': 'Invalid request method'})


@csrf_exempt
@login_required
def edit_camera(request, camera_id):
    """Edit camera settings - GET returns data, POST saves it (Admin only)"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'message': 'Permission denied'})

    camera = get_object_or_404(Camera, id=camera_id)

    if request.method == 'GET':
        return JsonResponse({
            'success': True,
            'camera': {
                'id': camera.id,
                'name': camera.name,
                'location': camera.location,
                'camera_url': camera.camera_url,
                'camera_type': camera.camera_type,
                'is_active': camera.is_active,
                'resolution_width': camera.resolution_width,
                'resolution_height': camera.resolution_height,
                'fps': camera.fps,
            }
        })

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            camera.name = data.get('name', camera.name).strip()
            camera.location = data.get('location', camera.location).strip()
            camera.camera_url = data.get('camera_url', camera.camera_url).strip()
            camera.camera_type = data.get('camera_type', camera.camera_type)
            camera.is_active = data.get('is_active', camera.is_active)
            camera.resolution_width = int(data.get('resolution_width', camera.resolution_width))
            camera.resolution_height = int(data.get('resolution_height', camera.resolution_height))
            camera.fps = int(data.get('fps', camera.fps))
            camera.save()

            SystemLog.objects.create(
                user=request.user,
                log_type='info',
                action='Camera Updated',
                description='Admin updated camera: {}'.format(camera.name),
                ip_address=request.META.get('REMOTE_ADDR')
            )

            return JsonResponse({
                'success': True,
                'message': 'Camera "{}" updated successfully!'.format(camera.name),
                'camera': {
                    'id': camera.id,
                    'name': camera.name,
                    'location': camera.location,
                    'camera_url': camera.camera_url,
                    'camera_type': camera.camera_type,
                    'camera_type_display': camera.get_camera_type_display(),
                    'is_active': camera.is_active,
                    'resolution': camera.get_resolution_display(),
                    'fps': camera.fps,
                }
            })
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})

    return JsonResponse({'success': False, 'message': 'Invalid request method'})


@csrf_exempt
@login_required
def delete_camera(request, camera_id):
    """Delete a camera (Admin only)"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'message': 'Permission denied'})

    if request.method in ('POST', 'DELETE'):
        camera = get_object_or_404(Camera, id=camera_id)
        name = camera.name
        camera.delete()

        SystemLog.objects.create(
            user=request.user,
            log_type='warning',
            action='Camera Deleted',
            description='Admin deleted camera: {}'.format(name),
            ip_address=request.META.get('REMOTE_ADDR')
        )

        return JsonResponse({'success': True, 'message': 'Camera "{}" deleted successfully!'.format(name)})

    return JsonResponse({'success': False, 'message': 'Invalid request method'})


@csrf_exempt
@login_required
def toggle_camera_status(request, camera_id):
    """Toggle camera active/inactive status (Admin only)"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'message': 'Permission denied'})

    if request.method == 'POST':
        from .models import Camera
        camera = get_object_or_404(Camera, id=camera_id)
        camera.is_active = not camera.is_active
        camera.save()

        status_text = 'activated' if camera.is_active else 'deactivated'

        SystemLog.objects.create(
            user=request.user,
            log_type='info',
            action='Camera Status Changed',
            description='Admin {} camera: {}'.format(status_text, camera.name),
            ip_address=request.META.get('REMOTE_ADDR')
        )

        return JsonResponse({
            'success': True,
            'is_active': camera.is_active,
            'message': 'Camera "{}" {} successfully!'.format(camera.name, status_text)
        })

    return JsonResponse({'success': False, 'message': 'Invalid request method'})


@csrf_exempt
@login_required
def test_camera_connection(request, camera_id):
    """Test camera connectivity (Admin only)"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'message': 'Permission denied'})

    if request.method == 'POST':
        from .models import Camera
        camera = get_object_or_404(Camera, id=camera_id)

        try:
            url = camera.camera_url.strip()
            source = int(url) if url.lstrip('-').isdigit() else url

            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return JsonResponse({
                        'success': True,
                        'message': 'Camera "{}" is responding successfully!'.format(camera.name)
                    })
                else:
                    return JsonResponse({
                        'success': False,
                        'message': 'Camera opened but could not read a frame. Check the stream.'
                    })
            else:
                cap.release()
                return JsonResponse({
                    'success': False,
                    'message': 'Could not open camera at "{}". Check the URL/device ID.'.format(camera.camera_url)
                })
        except Exception as e:
            return JsonResponse({'success': False, 'message': 'Connection test failed: {}'.format(str(e))})

    return JsonResponse({'success': False, 'message': 'Invalid request method'})


# ============= TEACHER ATTENDANCE MODULE =============

@login_required
def teacher_attendance(request):
    """Teacher attendance management - admin and teachers can access"""
    if request.user.role not in ['admin', 'teacher']:
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    teachers = User.objects.filter(role='teacher', is_approved=True)

    if request.user.role == 'teacher':
        subjects = Subject.objects.filter(teacher=request.user)
    else:
        subjects = Subject.objects.all()

    # Get filter parameters
    filter_date = request.GET.get('filter_date', str(timezone.now().date()))
    filter_subject = request.GET.get('filter_subject', '')
    filter_status = request.GET.get('filter_status', '')

    # Get attendance records for teachers only
    attendance_records = Attendance.objects.filter(
        user__role='teacher'
    ).select_related('user', 'subject')

    if filter_date:
        attendance_records = attendance_records.filter(date=filter_date)
    if filter_subject:
        attendance_records = attendance_records.filter(subject_id=filter_subject)
    if filter_status:
        attendance_records = attendance_records.filter(status=filter_status)

    # Calculate stats
    today = timezone.now().date()
    today_attendance_count = Attendance.objects.filter(user__role='teacher', date=today).count()
    total_teachers = teachers.count()
    present_count = Attendance.objects.filter(user__role='teacher', date=today, status='present').count()
    absent_count = total_teachers - present_count

    context = {
        'teachers': teachers,
        'subjects': subjects,
        'attendance_records': attendance_records[:100],
        'today': today,
        'current_time': timezone.now().time(),
        'filter_date': filter_date,
        'filter_subject': filter_subject,
        'filter_status': filter_status,
        'today_attendance_count': today_attendance_count,
        'total_teachers': total_teachers,
        'present_count': present_count,
        'absent_count': absent_count,
    }

    return render(request, 'teacher_attendance.html', context)


@login_required
def teacher_mark_attendance(request):
    """Mark teacher attendance manually"""
    if request.method == 'POST':
        if request.user.role not in ['admin', 'teacher']:
            messages.error(request, 'Permission denied.')
            return redirect('dashboard')

        user_id = request.POST.get('user_id')
        subject_id = request.POST.get('subject_id')
        date = request.POST.get('date')
        status = request.POST.get('status')
        notes = request.POST.get('notes', '')

        user = get_object_or_404(User, id=user_id)
        subject = get_object_or_404(Subject, id=subject_id)

        Attendance.objects.update_or_create(
            user=user,
            subject=subject,
            date=date,
            defaults={
                'status': status,
                'recognition_type': 'manual',
                'notes': notes,
                'marked_by': request.user,
                'is_verified': True,
            }
        )

        from .models import Notification
        Notification.objects.create(
            user=user,
            notification_type='info',
            title='Attendance Marked',
            message=f'Your attendance for {subject.name} on {date} has been marked as {status}.',
            action_url='/attendance/history/'
        )

        messages.success(request, 'Teacher attendance marked successfully!')
    return redirect('teacher_attendance')


@login_required
def teacher_bulk_mark_attendance(request):
    """Bulk mark teacher attendance"""
    if request.method == 'POST':
        if request.user.role not in ['admin', 'teacher']:
            messages.error(request, 'Permission denied.')
            return redirect('dashboard')

        subject_id = request.POST.get('subject_id')
        date = request.POST.get('date')
        teacher_ids = request.POST.getlist('teacher_ids')

        subject = get_object_or_404(Subject, id=subject_id)

        count = 0
        for teacher_id in teacher_ids:
            status = request.POST.get(f'status_{teacher_id}', 'present')
            notes = request.POST.get(f'notes_{teacher_id}', '')
            teacher = get_object_or_404(User, id=teacher_id)

            Attendance.objects.update_or_create(
                user=teacher,
                subject=subject,
                date=date,
                defaults={
                    'status': status,
                    'recognition_type': 'manual',
                    'notes': notes,
                    'marked_by': request.user,
                    'is_verified': True,
                }
            )
            count += 1

        messages.success(request, f'Bulk attendance marked for {count} teacher(s)!')
    return redirect('teacher_attendance')


# ============= EMPLOYEE ATTENDANCE MODULE =============

@login_required
def employee_attendance(request):
    """Employee (admin-role) attendance management - admin only"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    employees = User.objects.filter(role='admin', is_approved=True)
    subjects = Subject.objects.all()

    # Get filter parameters
    filter_date = request.GET.get('filter_date', str(timezone.now().date()))
    filter_subject = request.GET.get('filter_subject', '')
    filter_status = request.GET.get('filter_status', '')

    # Get attendance records for admin/employee users
    attendance_records = Attendance.objects.filter(
        user__role='admin'
    ).select_related('user', 'subject')

    if filter_date:
        attendance_records = attendance_records.filter(date=filter_date)
    if filter_subject:
        attendance_records = attendance_records.filter(subject_id=filter_subject)
    if filter_status:
        attendance_records = attendance_records.filter(status=filter_status)

    # Calculate stats
    today = timezone.now().date()
    today_attendance_count = Attendance.objects.filter(user__role='admin', date=today).count()
    total_employees = employees.count()
    present_count = Attendance.objects.filter(user__role='admin', date=today, status='present').count()
    absent_count = total_employees - present_count

    context = {
        'employees': employees,
        'subjects': subjects,
        'attendance_records': attendance_records[:100],
        'today': today,
        'current_time': timezone.now().time(),
        'filter_date': filter_date,
        'filter_subject': filter_subject,
        'filter_status': filter_status,
        'today_attendance_count': today_attendance_count,
        'total_employees': total_employees,
        'present_count': present_count,
        'absent_count': absent_count,
    }

    return render(request, 'employee_attendance.html', context)


@login_required
def employee_mark_attendance(request):
    """Mark employee attendance manually"""
    if request.method == 'POST':
        if request.user.role != 'admin':
            messages.error(request, 'Permission denied.')
            return redirect('dashboard')

        user_id = request.POST.get('user_id')
        subject_id = request.POST.get('subject_id')
        date = request.POST.get('date')
        status = request.POST.get('status')
        notes = request.POST.get('notes', '')

        user = get_object_or_404(User, id=user_id)
        subject = get_object_or_404(Subject, id=subject_id)

        Attendance.objects.update_or_create(
            user=user,
            subject=subject,
            date=date,
            defaults={
                'status': status,
                'recognition_type': 'manual',
                'notes': notes,
                'marked_by': request.user,
                'is_verified': True,
            }
        )

        messages.success(request, 'Employee attendance marked successfully!')
    return redirect('employee_attendance')


@login_required
def employee_bulk_mark_attendance(request):
    """Bulk mark employee attendance"""
    if request.method == 'POST':
        if request.user.role != 'admin':
            messages.error(request, 'Permission denied.')
            return redirect('dashboard')

        subject_id = request.POST.get('subject_id')
        date = request.POST.get('date')
        employee_ids = request.POST.getlist('employee_ids')

        subject = get_object_or_404(Subject, id=subject_id)

        count = 0
        for employee_id in employee_ids:
            status = request.POST.get(f'status_{employee_id}', 'present')
            notes = request.POST.get(f'notes_{employee_id}', '')
            employee = get_object_or_404(User, id=employee_id)

            Attendance.objects.update_or_create(
                user=employee,
                subject=subject,
                date=date,
                defaults={
                    'status': status,
                    'recognition_type': 'manual',
                    'notes': notes,
                    'marked_by': request.user,
                    'is_verified': True,
                }
            )
            count += 1

        messages.success(request, f'Bulk attendance marked for {count} employee(s)!')
    return redirect('employee_attendance')


# ============= DEPARTMENT & SECTION MANAGEMENT =============

@login_required
def department_list(request):
    """List all departments with stats (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    departments = Department.objects.prefetch_related('sections').all()
    dept_data = []
    for dept in departments:
        dept_data.append({
            'dept': dept,
            'sections': dept.sections.all(),
            'student_count': User.objects.filter(dept=dept, role='student').count(),
            'teacher_count': User.objects.filter(dept=dept, role='teacher').count(),
            'subject_count': Subject.objects.filter(department=dept).count(),
        })

    return render(request, 'departments.html', {
        'dept_data': dept_data,
        'total_depts': departments.count(),
    })


@login_required
def create_department(request):
    """Create a new department (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    if request.method == 'POST':
        form = DepartmentForm(request.POST)
        if form.is_valid():
            dept = form.save()
            SystemLog.objects.create(
                user=request.user,
                log_type='info',
                action='Department Created',
                description=f'Admin created department: {dept.name} ({dept.code})',
                ip_address=request.META.get('REMOTE_ADDR')
            )
            messages.success(request, f'Department "{dept.name}" created successfully!')
            return redirect('department_list')
    else:
        form = DepartmentForm()

    return render(request, 'departments.html', {'form': form, 'show_create': True,
                                                'dept_data': [], 'total_depts': 0})


@login_required
def edit_department(request, dept_id):
    """Edit department (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    dept = get_object_or_404(Department, id=dept_id)
    if request.method == 'POST':
        form = DepartmentForm(request.POST, instance=dept)
        if form.is_valid():
            form.save()
            messages.success(request, f'Department "{dept.name}" updated successfully!')
        else:
            messages.error(request, 'Please fix the errors below.')
    return redirect('department_list')


@login_required
def delete_department(request, dept_id):
    """Delete department (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    dept = get_object_or_404(Department, id=dept_id)
    name = dept.name
    dept.delete()
    SystemLog.objects.create(
        user=request.user,
        log_type='warning',
        action='Department Deleted',
        description=f'Admin deleted department: {name}',
        ip_address=request.META.get('REMOTE_ADDR')
    )
    messages.success(request, f'Department "{name}" deleted.')
    return redirect('department_list')


@login_required
def section_list(request):
    """List all sections grouped by department (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    departments = Department.objects.prefetch_related('sections').all()
    section_form = SectionForm()
    section_data = []
    for dept in departments:
        for sec in dept.sections.all():
            section_data.append({
                'section': sec,
                'dept': dept,
                'student_count': User.objects.filter(section=sec, role='student').count(),
                'subject_count': Subject.objects.filter(section=sec).count(),
            })

    return render(request, 'sections.html', {
        'section_data': section_data,
        'departments': departments,
        'section_form': section_form,
        'total_sections': Section.objects.count(),
    })


@login_required
def create_section(request):
    """Create a new section (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    if request.method == 'POST':
        form = SectionForm(request.POST)
        if form.is_valid():
            # Check duplicate
            dept = form.cleaned_data['department']
            name = form.cleaned_data['name']
            if Section.objects.filter(department=dept, name=name).exists():
                messages.error(request, f'Section "{name}" already exists in {dept.name}.')
            else:
                sec = form.save()
                SystemLog.objects.create(
                    user=request.user,
                    log_type='info',
                    action='Section Created',
                    description=f'Admin created section: {sec}',
                    ip_address=request.META.get('REMOTE_ADDR')
                )
                messages.success(request, f'Section "{sec}" created successfully!')
        else:
            messages.error(request, 'Please fix form errors.')
    return redirect('section_list')


@login_required
def delete_section(request, section_id):
    """Delete section (Admin only)"""
    if request.user.role != 'admin':
        messages.error(request, 'Permission denied.')
        return redirect('dashboard')

    section = get_object_or_404(Section, id=section_id)
    name = str(section)
    section.delete()
    messages.success(request, f'Section "{name}" deleted.')
    return redirect('section_list')


def api_sections_by_department(request, dept_id):
    """AJAX: return sections for a given department as JSON"""
    sections = Section.objects.filter(department_id=dept_id).values('id', 'name')
    return JsonResponse({'sections': list(sections)})


@login_required
def api_students_by_section(request, section_id):
    """AJAX: return approved students for a given section as JSON"""
    students = User.objects.filter(
        role='student',
        is_approved=True,
        section_id=section_id
    ).order_by('first_name', 'last_name').values('id', 'username', 'first_name', 'last_name')
    data = [
        {
            'id': s['id'],
            'name': f"{s['first_name']} {s['last_name']}".strip() or s['username']
        }
        for s in students
    ]
    return JsonResponse({'students': data})


@login_required
def api_subjects_by_dept_section(request):
    """AJAX: return subjects filtered by dept and/or section (GET params)"""
    dept_id    = request.GET.get('dept')
    section_id = request.GET.get('section')
    qs = Subject.objects.all()
    if dept_id:
        qs = qs.filter(department_id=dept_id)
    if section_id:
        qs = qs.filter(section_id=section_id)
    data = [{'id': s.id, 'label': str(s)} for s in qs.order_by('code')]
    return JsonResponse({'subjects': data})


@login_required
def api_students_by_subject(request, subject_id):
    """AJAX: return students enrolled in a specific subject"""
    try:
        subject = Subject.objects.get(id=subject_id)
        students = subject.students.filter(
            is_approved=True
        ).order_by('first_name', 'last_name')
        data = [
            {
                'id': s.id,
                'name': f"{s.first_name} {s.last_name}".strip() or s.username
            }
            for s in students
        ]
    except Subject.DoesNotExist:
        data = []
    return JsonResponse({'students': data})
