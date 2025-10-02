from rest_framework import serializers
from .models import RiceInfo

class RiceInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = RiceInfo
        fields = ['variety_name', 'info']

class PredictionSerializer(serializers.Serializer):
    rice_image = serializers.ImageField()
