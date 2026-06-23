from .models import LiveAttendanceSession


def live_session_context(request):
    """Inject live_session_active into every template for authenticated students."""
    if request.user.is_authenticated and hasattr(request.user, 'role') and request.user.role == 'student':
        active = LiveAttendanceSession.objects.filter(
            is_active=True,
            subject__students=request.user
        ).exists()
        return {'live_session_active': active}
    return {'live_session_active': False}
