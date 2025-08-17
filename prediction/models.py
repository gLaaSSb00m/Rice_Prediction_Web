from django.db import models

class RiceInfo(models.Model):
    """Model to store information about different rice varieties."""
    
    variety_name = models.CharField(
        max_length=100, 
        unique=True,
        help_text="Name of the rice variety"
    )
    
    info = models.TextField(
        default="info isn't available",
        help_text="Detailed information about this rice variety"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Rice Information"
        verbose_name_plural = "Rice Information"
        ordering = ['variety_name']
    
    def __str__(self):
        return f"{self.variety_name}"
