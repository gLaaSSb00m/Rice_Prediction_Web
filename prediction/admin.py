from django.contrib import admin
from .models import RiceInfo, RiceModel

@admin.register(RiceInfo)
class RiceInfoAdmin(admin.ModelAdmin):
    list_display = ['variety_name', 'created_at', 'updated_at']
    search_fields = ['variety_name']
    list_filter = ['created_at']
    ordering = ['variety_name']

@admin.register(RiceModel)
class RiceModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active', 'created_at', 'updated_at']
    search_fields = ['name']
    list_filter = ['is_active', 'created_at']
    ordering = ['-created_at']
