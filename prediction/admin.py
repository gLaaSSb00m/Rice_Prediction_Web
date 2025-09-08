from django.contrib import admin
from .models import RiceInfo

@admin.register(RiceInfo)
class RiceInfoAdmin(admin.ModelAdmin):
    list_display = ['variety_name', 'created_at', 'updated_at']
    search_fields = ['variety_name']
    list_filter = ['created_at']
    ordering = ['variety_name']
