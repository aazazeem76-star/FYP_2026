from django.apps import AppConfig
import time


# Stamp the exact second this process started.
# SessionBootMiddleware compares every incoming session against this value
# and flushes any session that was created before this boot.
SERVER_BOOT_TIME = time.time()

# Flag so we only flush sessions once (on the first request after boot).
_sessions_flushed = False


class AttendanceAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'attendance_app'

    def ready(self):
        """
        Connect a one-shot signal that purges ALL existing database sessions
        on the very first HTTP request after each server (re)start.

        Why signal instead of direct DB call here:
          Django discourages DB access inside ready(); doing it inside a
          request_started handler is the idiomatic safe approach.
        """
        from django.core.signals import request_started
        request_started.connect(_flush_sessions_once)


def _flush_sessions_once(sender, **kwargs):
    """
    Delete every django_session row exactly once per server boot.

    Context: SESSION_EXPIRE_AT_BROWSER_CLOSE = True relies on the browser
    discarding its session cookie when it closes.  Modern browsers restore
    tabs/cookies after a restart, so stale cookies let users skip the login
    page.  By deleting all session rows at startup, those cookies become
    invalid on the next request and users are redirected to the home page.
    """
    global _sessions_flushed
    if _sessions_flushed:
        return
    _sessions_flushed = True

    # Disconnect immediately so this never runs again in this process
    from django.core.signals import request_started
    request_started.disconnect(_flush_sessions_once)

    try:
        from django.contrib.sessions.models import Session
        deleted, _ = Session.objects.all().delete()
        if deleted:
            print(f'[FRA] {deleted} stale session(s) cleared on startup. '
                  'All users must log in again.')
    except Exception as exc:
        # Table may not exist yet on a fresh migrate — safe to ignore
        print(f'[FRA] Session flush skipped: {exc}')
