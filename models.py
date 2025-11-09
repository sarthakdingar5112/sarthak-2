from django.db import models
from django.contrib.auth.models import User

LANG_CHOICES = [
    ('te', 'Telugu'),
    ('hi', 'Hindi'),
    ('en', 'English'),
    ('ur', 'Urdu'),
    ('ml', 'Malayalam'),
    ('ta', 'Tamil'),
    ('kn', 'Kannada'),
]

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    preferred_language = models.CharField(max_length=2, choices=LANG_CHOICES, default='en')
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username

class Prompt(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    text = models.TextField()
    language = models.CharField(max_length=2, choices=LANG_CHOICES, default='en')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return (self.text or '')[:60]

class Analysis(models.Model):
    prompt = models.OneToOneField(Prompt, on_delete=models.CASCADE, related_name='analysis')
    top_moods = models.JSONField(default=list)
    multi_moods = models.JSONField(default=list)
    sentiment_label = models.CharField(max_length=20)
    sentiment_score = models.FloatField(default=0.0)
    spotify_tracks = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

class AudioBlob(models.Model):
    filename = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100, default='audio/wav')
    data = models.BinaryField()
    size = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename

class Generation(models.Model):
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE, related_name='generations')
    duration_s = models.PositiveIntegerField(default=20)
    model_name = models.CharField(max_length=100, default='facebook/musicgen-small')
    temperature = models.FloatField(default=1.0)
    top_k = models.IntegerField(null=True, blank=True)
    top_p = models.FloatField(null=True, blank=True)
    guidance_scale = models.FloatField(default=3.0)
    seed = models.IntegerField(null=True, blank=True)
    metadata_json = models.JSONField(default=dict, blank=True)
    audio_blob = models.ForeignKey(
        AudioBlob, null=True, blank=True, on_delete=models.SET_NULL, related_name='generations'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Gen #{self.id} for Prompt {self.prompt_id} ({self.duration_s}s)"

class UploadedImage(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    image = models.ImageField(upload_to='uploaded_images/')
    predicted_moods = models.JSONField(default=list)  # [(label, score), ...]
    spotify_tracks = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
