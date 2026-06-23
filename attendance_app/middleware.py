"""
SessionBootMiddleware
--------------------
Invalidates any session whose '_boot_time' stamp predates the current
server start.  This catches the case where a browser restores a stale
session cookie after the machine reboots or the dev-server restarts,
which would otherwise bypass the login page entirely.
"""

from django.shortcuts import redirect
from django.urls import reverse
from attendance_app.apps import SERVER_BOOT_TIME


class SessionBootMiddleware:
    """
    For every request that carries an authenticated session:
      1. Read the '_boot_time' we embedded when the session was created.
      2. If that stamp is older than SERVER_BOOT_TIME (i.e. it was created
         before the current process started), flush the session and redirect
         the user to the home/login page.
      3. Otherwise stamp the session with the current boot time so it
         survives future requests within the same server run.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Only act on requests that already have a session
        if hasattr(request, 'session') and request.session.session_key:
            # Use None as sentinel — distinguishes:
            #   None  → brand-new session created THIS server run (just stamp it)
            #   0     → old session from BEFORE this server run (flush it)
            #   value → valid session from this run (refresh stamp)
            session_boot = request.session.get('_boot_time', None)

            if session_boot is None:
                # New session created in this server run (e.g. just after login).
                # Stamp it so future requests pass through cleanly.
                request.session['_boot_time'] = SERVER_BOOT_TIME
                request.session.modified = True

            elif session_boot < SERVER_BOOT_TIME:
                # Stale session left over from a PREVIOUS server run.
                # The all-session flush in apps.py normally removes these on
                # the first request, but this guard catches any that slip through
                # (e.g. the session was created between the flush and now).
                request.session.flush()
                # Only redirect non-public paths — login/register/home are fine.
                safe_paths = {
                    reverse('index'),
                    reverse('login'),
                    reverse('register'),
                    reverse('verify_otp'),
                }
                if request.path not in safe_paths:
                    return redirect('index')

            else:
                # Valid session from this server run — refresh the stamp.
                request.session['_boot_time'] = SERVER_BOOT_TIME
                request.session.modified = True

        response = self.get_response(request)
        return response
