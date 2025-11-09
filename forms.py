from django import forms
from django.contrib.auth.models import User
from .models import LANG_CHOICES

# --------- Auth forms ---------

class RegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    preferred_language = forms.ChoiceField(choices=LANG_CHOICES, initial='en')
    avatar = forms.ImageField(required=False)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']


class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)


# --------- Text analyze + generation ---------

class AnalyzeForm(forms.Form):
    prompt = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': 'Describe the music...'})
    )
    language = forms.ChoiceField(choices=LANG_CHOICES, initial='en', label='Language')
    do_spotify = forms.BooleanField(required=False, initial=True, label='Search Spotify for similar tracks')


class GenerateForm(forms.Form):
    prompt_id = forms.IntegerField(widget=forms.HiddenInput())
    duration = forms.IntegerField(min_value=5, initial=15, help_text='Seconds')
    model = forms.CharField(initial='facebook/musicgen-small')
    temperature = forms.FloatField(initial=1.0)
    top_k = forms.IntegerField(required=False)
    top_p = forms.FloatField(required=False, initial=0.95)
    guidance_scale = forms.FloatField(initial=3.0)
    seed = forms.IntegerField(required=False)


# --------- Image analyze ---------

class ImageAnalyzeForm(forms.Form):
    image = forms.ImageField()
    language = forms.ChoiceField(choices=LANG_CHOICES, initial='en', label='Language')
    do_spotify = forms.BooleanField(required=False, initial=True)
