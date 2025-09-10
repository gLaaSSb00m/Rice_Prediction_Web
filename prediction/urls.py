from django.urls import path
from . import views
from .api_views import PredictAPIView, RiceInfoListAPIView

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('api/predict/', PredictAPIView.as_view(), name='api_predict'),
    path('api/riceinfo/', RiceInfoListAPIView.as_view(), name='api_riceinfo_list'),
]
