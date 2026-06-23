import django, os, sys, cv2, numpy as np
os.environ['DJANGO_SETTINGS_MODULE'] = 'attendance_project.settings'
sys.path.insert(0, r'e:\backup05042026')
django.setup()

from attendance_app.models import BiometricSample
from django.contrib.auth import get_user_model
from django.conf import settings as dj_settings
from ai_models.face_recognition import FaceRecognition

User = get_user_model()
fr = FaceRecognition()

# Load the actual live frame that was captured
live_path = os.path.join(str(dj_settings.MEDIA_ROOT), 'debug_live_frame.jpg')
live_img = cv2.imread(live_path)
if live_img is None:
    print('ERROR: debug_live_frame.jpg not found. Try verifying again first.')
    sys.exit(1)

print(f'Live frame: shape={live_img.shape}')
faces = fr.detect_faces(live_img)
print(f'Faces in live frame: {len(faces)}')
for f in faces:
    print(f'  face box: {f}')

live_desc = fr.extract_combined_encoding(live_img)
live_flip = fr.extract_combined_encoding(cv2.flip(live_img, 1))
print(f'live_desc: {"OK" if live_desc is not None else "NONE"}')
print(f'live_flip: {"OK" if live_flip is not None else "NONE"}')

# Load teacher samples
teacher = User.objects.filter(role='teacher').first()
samples = list(BiometricSample.objects.filter(user=teacher, sample_type='face'))
print(f'\nTeacher: {teacher.username}, {len(samples)} samples')

user_best = 0.0
print('\n=== Live vs Teacher stored samples ===')
for i, s in enumerate(samples):
    p = os.path.join(str(dj_settings.MEDIA_ROOT), s.image_path)
    img = cv2.imread(p)
    if img is None:
        print(f'  [{i}] FAILED to load')
        continue
    stored_desc = fr.extract_combined_encoding(img)
    if stored_desc is None:
        print(f'  [{i}] stored_desc = NONE')
        continue
    s1 = fr.cosine_similarity(live_desc, stored_desc) if live_desc is not None else 0.0
    s2 = fr.cosine_similarity(live_flip, stored_desc) if live_flip is not None else 0.0
    best = max(s1, s2)
    user_best = max(user_best, best)
    print(f'  [{i}] normal={s1:.4f}  flipped={s2:.4f}  best={best:.4f}')

print(f'\n  --> user_best = {user_best:.4f}')

# Load other user samples
others = list(BiometricSample.objects.filter(sample_type='face').exclude(user=teacher))
print(f'\n=== Live vs Other users ({len(others)} samples) ===')
other_best = 0.0
for s in others:
    p = os.path.join(str(dj_settings.MEDIA_ROOT), s.image_path)
    img = cv2.imread(p)
    if img is None:
        continue
    stored_desc = fr.extract_combined_encoding(img)
    if stored_desc is None:
        continue
    s1 = fr.cosine_similarity(live_desc, stored_desc) if live_desc is not None else 0.0
    s2 = fr.cosine_similarity(live_flip, stored_desc) if live_flip is not None else 0.0
    best = max(s1, s2)
    other_best = max(other_best, best)
    print(f'  {s.user.username}: normal={s1:.4f}  flipped={s2:.4f}  best={best:.4f}')

print(f'\n  --> other_best = {other_best:.4f}')

print(f'\n=== GATE CHECK (current thresholds) ===')
print(f'  user_best    = {user_best:.4f}  (need >= 0.78)')
print(f'  other_best   = {other_best:.4f}')
print(f'  margin       = {user_best - other_best:.4f}  (need >= 0.05 OR solo_floor 0.80)')
if other_best == 0.0:
    gate = user_best >= 0.80
    print(f'  SOLO_FLOOR gate (0.80): {gate}')
else:
    gate = (user_best - other_best) >= 0.05
    print(f'  NEGATIVE_MARGIN gate:   {gate}')
print(f'  BEST_SCORE gate (0.78): {user_best >= 0.78}')
print(f'\n  WOULD PASS: {user_best >= 0.78 and gate}')
