from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('get_model/', views.get_model, name='get_model'),
    path('get_rice_info/', views.get_rice_info, name='get_rice_info'),
]
