from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("studio", "0002_generation_agent")]

    operations = [
        migrations.AddField(
            model_name="generation",
            name="duration",
            field=models.PositiveSmallIntegerField(default=5),
        ),
    ]
