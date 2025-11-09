# musicsite/musicgen/views.py

import json
import numpy as np
from io import BytesIO
from PIL import Image

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .forms import (
    RegisterForm, LoginForm,
    AnalyzeForm, GenerateForm, ImageAnalyzeForm
)
from .models import (
    UserProfile, Prompt, Analysis, Generation, AudioBlob, UploadedImage
)
from .services import analyze_prompt, generate_audio


# =========================
# AUTH PAGES
# =========================

@require_http_methods(["GET", "POST"])
def register_view(request):
    """Register a new user and profile."""
    if request.method == 'POST':
        form = RegisterForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save(commit=False)
            password = form.cleaned_data['password']
            user.set_password(password)
            user.save()

            UserProfile.objects.create(
                user=user,
                preferred_language=form.cleaned_data['preferred_language'],
                avatar=form.cleaned_data.get('avatar')
            )
            messages.success(request, "Account created successfully! Please log in.")
            return redirect('musicgen:login')
    else:
        form = RegisterForm()
    return render(request, 'musicgen/register.html', {'form': form})


@require_http_methods(["GET", "POST"])
def login_view(request):
    """User login."""
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(
                request,
                username=form.cleaned_data['username'],
                password=form.cleaned_data['password']
            )
            if user:
                login(request, user)
                return redirect('musicgen:home')
            messages.error(request, "Invalid credentials.")
    else:
        form = LoginForm()
    return render(request, 'musicgen/login.html', {'form': form})


def logout_view(request):
    """Logout and go to login."""
    logout(request)
    return redirect('musicgen:login')


# =========================
# PAGES
# =========================

@login_required
def home(request):
    """Dashboard shown after login."""
    prompts = Prompt.objects.filter(user=request.user).order_by('-created_at')[:20]
    return render(request, 'musicgen/home.html', {'prompts': prompts})


# ---------- Text analyze ----------

@require_http_methods(["GET", "POST"])
@login_required
def analyze_view(request):
    """Analyze mood from text; saves Prompt + Analysis."""
    if request.method == 'POST':
        form = AnalyzeForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['prompt']
            language = form.cleaned_data['language']
            do_spotify = form.cleaned_data['do_spotify']

            prompt = Prompt.objects.create(
                user=request.user,
                text=text,
                language=language,
            )

            data = analyze_prompt(f"{text} language:{language}", do_spotify=do_spotify)
            Analysis.objects.create(
                prompt=prompt,
                top_moods=data['top_moods'],
                multi_moods=data['multi_moods'],
                sentiment_label=data['sentiment_label'],
                sentiment_score=data['sentiment_score'],
                spotify_tracks=data['spotify_tracks'],
            )
            return redirect('musicgen:prompt_detail', pk=prompt.pk)
    else:
        initial_lang = getattr(getattr(request.user, 'profile', None), 'preferred_language', 'en')
        form = AnalyzeForm(initial={'language': initial_lang})

    return render(request, 'musicgen/analyze.html', {'form': form})


@login_required
def prompt_detail(request, pk):
    """Show analysis results + generation form for a prompt."""
    prompt = get_object_or_404(Prompt, pk=pk, user=request.user)
    gen_form = GenerateForm(initial={'prompt_id': prompt.id})
    return render(request, 'musicgen/detail.html', {'prompt': prompt, 'gen_form': gen_form})


@require_http_methods(["POST"])
@login_required
def generate_view(request):
    """Generate audio for a prompt; stores audio in DB as BinaryField."""
    form = GenerateForm(request.POST)
    if not form.is_valid():
        pid = request.POST.get('prompt_id')
        prompt = get_object_or_404(Prompt, pk=pid, user=request.user) if pid else None
        gen_form = GenerateForm(initial={'prompt_id': pid})
        return render(request, 'musicgen/detail.html', {
            'prompt': prompt,
            'gen_form': gen_form,
            'errors': form.errors
        })

    prompt = get_object_or_404(Prompt, pk=form.cleaned_data['prompt_id'], user=request.user)

    buf, sr, meta = generate_audio(
        prompt=prompt.text,
        duration=form.cleaned_data['duration'],
        model_name=form.cleaned_data['model'],
        temperature=form.cleaned_data['temperature'],
        top_k=form.cleaned_data.get('top_k'),
        top_p=form.cleaned_data.get('top_p'),
        guidance_scale=form.cleaned_data['guidance_scale'],
        seed=form.cleaned_data.get('seed'),
    )

    wav_bytes = buf.read()
    blob = AudioBlob.objects.create(
        filename=f"music_{prompt.id}.wav",
        content_type='audio/wav',
        data=wav_bytes,
        size=len(wav_bytes),
    )

    gen = Generation.objects.create(
        prompt=prompt,
        duration_s=meta['duration_s'],
        model_name=meta['model_name'],
        temperature=meta['temperature'],
        top_k=meta['top_k'],
        top_p=meta['top_p'],
        guidance_scale=meta['guidance_scale'],
        seed=meta['seed'],
        metadata_json=meta,
        audio_blob=blob,
    )

    return redirect('musicgen:generation_detail', pk=gen.pk)


@login_required
def generation_detail(request, pk):
    """Show generated audio and info, with <audio> player in template."""
    gen = get_object_or_404(Generation, pk=pk, prompt__user=request.user)
    return render(request, 'musicgen/generation_detail.html', {'gen': gen})


@login_required
def audio_blob_view(request, pk):
    """Serve generated audio as inline WAV from DB."""
    blob = get_object_or_404(AudioBlob, pk=pk)
    resp = HttpResponse(blob.data, content_type=blob.content_type or 'audio/wav')
    resp['Content-Length'] = str(blob.size or len(blob.data))
    resp['Content-Disposition'] = f'inline; filename="{blob.filename}"'
    return resp


# ---------- Image analyze ----------

@require_http_methods(["GET", "POST"])
@login_required
def image_analyze_view(request):
    """Heuristic image → moods + Spotify suggestions."""
    if request.method == 'POST':
        form = ImageAnalyzeForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            language = form.cleaned_data['language']
            do_spotify = form.cleaned_data['do_spotify']

            # Fast color heuristic (CPU friendly)
            image = Image.open(img_file).convert('RGB').resize((224, 224))
            np_img = np.array(image).astype(np.float32) / 255.0

            r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
            sat = np.std(np.stack([r, g, b], axis=-1), axis=-1).mean()
            bright = np.mean(np_img)

            if bright > 0.6 and sat > 0.12:
                moods = [('joy', 0.85), ('energetic', 0.65)]
            elif bright < 0.35 and sat < 0.10:
                moods = [('melancholic', 0.80), ('sad', 0.60)]
            elif sat > 0.15:
                moods = [('romantic', 0.70), ('joy', 0.55)]
            else:
                moods = [('relaxed', 0.70), ('calm', 0.55)]

            # Get songs using top mood + language
            top_label = moods[0][0]
            prompt_like = f"{top_label} {language} song"
            data = analyze_prompt(prompt_like, do_spotify=do_spotify)
            tracks = data['spotify_tracks']

            up = UploadedImage.objects.create(
                user=request.user,
                image=img_file,
                predicted_moods=moods,
                spotify_tracks=tracks,
            )
            return render(request, 'musicgen/image_result.html', {'img': up})
    else:
        initial_lang = getattr(getattr(request.user, 'profile', None), 'preferred_language', 'en')
        form = ImageAnalyzeForm(initial={'language': initial_lang})
    return render(request, 'musicgen/image_analyze.html', {'form': form})


# =========================
# JSON APIs
# =========================

@csrf_exempt
@require_http_methods(["POST"])
def api_register(request):
    """Register via JSON API."""
    try:
        payload = json.loads(request.body.decode('utf-8'))
        username = payload.get('username')
        email = payload.get('email', '')
        password = payload.get('password')
        preferred_language = payload.get('preferred_language', 'en')

        if not username or not password:
            return JsonResponse({'error': 'username and password required'}, status=400)

        from django.contrib.auth.models import User
        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'username already exists'}, status=409)

        user = User.objects.create_user(username=username, email=email, password=password)
        UserProfile.objects.create(user=user, preferred_language=preferred_language)
        return JsonResponse({'ok': True, 'user_id': user.id})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def api_login(request):
    """Login via JSON API (session-based)."""
    try:
        payload = json.loads(request.body.decode('utf-8'))
        username = payload.get('username')
        password = payload.get('password')
        user = authenticate(request, username=username, password=password)
        if not user:
            return JsonResponse({'error': 'invalid credentials'}, status=401)
        login(request, user)
        return JsonResponse({'ok': True, 'user_id': user.id})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_history(request, user_id: int):
    """Return user's text prompts + generations."""
    prompts = Prompt.objects.filter(user_id=user_id).order_by('-created_at')[:50]
    data = []
    for p in prompts:
        data.append({
            'prompt_id': p.id,
            'text': p.text,
            'language': p.language,
            'created_at': p.created_at.isoformat(),
            'analysis': p.analysis.top_moods if hasattr(p, 'analysis') else None,
            'generations': [
                {
                    'gen_id': g.id,
                    'duration': g.duration_s,
                    'created_at': g.created_at.isoformat(),
                }
                for g in p.generations.all().order_by('-created_at')
            ],
        })
    return JsonResponse({'ok': True, 'history': data})
