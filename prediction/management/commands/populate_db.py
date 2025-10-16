from django.core.management.base import BaseCommand
from prediction.models import RiceInfo, RiceModel
import os
from django.core.files import File
from django.conf import settings

RICE_CLASSES = [
    "10_Lal_Aush","11_Jirashail","12_Gutisharna","13_Red_Cargo","14_Najirshail",
    "15_Katari_Polao","16_Lal_Biroi","17_Chinigura_Polao","18_Amondhan","19_Shorna5",
    "1_Subol_Lota","20_Lal_Binni","21_Arborio","22_Turkish_Basmati","23_Ipsala",
    "24_Jasmine","25_Karacadag","26_BD30","27_BD33","28_BD39","29_BD49",
    "2_Bashmoti","30_BD51","31_BD52","32_BD56","33_BD57","34_BD70","35_BD72",
    "36_BD75","37_BD76","38_BD79","39_BD85","3_Ganjiya","40_BD87","41_BD91",
    "42_BD93","43_BD95","44_Binadhan7","45_Binadhan8","46_Binadhan10","47_Binadhan11",
    "48_Binadhan12","49_Binadhan14","4_Shampakatari","50_Binadhan16","51_Binadhan17",
    "52_Binadhan19","53_Binadhan21","54_Binadhan23","55_Binadhan24","56_Binadhan25",
    "57_Binadhan26","58_BR22","59_BR23","5_Katarivog","60_BRRI67","61_BRRI74",
    "62_BRRI102","6_BR28","7_BR29","8_Paijam","9_Bashful"
]

class Command(BaseCommand):
    help = 'Populate the database with rice classes and model file'

    def handle(self, *args, **options):
        # Populate RiceInfo with classes
        for class_name in RICE_CLASSES:
            RiceInfo.objects.get_or_create(
                variety_name=class_name,
                defaults={'info': 'Information not available.'}
            )
        self.stdout.write(self.style.SUCCESS(f'Populated {len(RICE_CLASSES)} rice varieties.'))

        # Populate RiceModel with the .h5 files
        models_to_add = [
            ('VGG16 Rice Classifier', 'best_VGG16_stage2.weights.h5'),
            ('MobileNetV2 Rice Classifier', 'MobileNetV2_rice62_final.weights.h5'),
            ('XGBoost Meta Model', 'xgb_meta_model.json'),
        ]

        for model_name, file_name in models_to_add:
            model_path = os.path.join(settings.BASE_DIR, 'models', file_name)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    file_obj = File(f, name=file_name)
                    model, created = RiceModel.objects.get_or_create(
                        name=model_name,
                        defaults={'model_file': file_obj, 'is_active': True}
                    )
                    if created:
                        self.stdout.write(self.style.SUCCESS(f'Created RiceModel: {model_name}'))
                    else:
                        self.stdout.write(self.style.SUCCESS(f'RiceModel {model_name} already exists.'))
            else:
                self.stdout.write(self.style.ERROR(f'Model file not found at {model_path}'))
