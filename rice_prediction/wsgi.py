<<<<<<< HEAD
"""
WSGI config for rice_prediction project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

=======
>>>>>>> 50fd73965494c71073b83d2ccf2e5a21042bd94f
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rice_prediction.settings')

application = get_wsgi_application()
