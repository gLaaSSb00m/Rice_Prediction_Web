# Generated manually for tflite_file field

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0002_ricemodel'),
    ]

    operations = [
        migrations.AddField(
            model_name='ricemodel',
            name='tflite_file',
            field=models.FileField(blank=True, help_text='Path to the .tflite model file', null=True, upload_to='models/'),
        ),
    ]
