# musicsite/musicgen/services.py

import os
import io
import json
import numpy as np
import soundfile as sf

# Keep Spotify optional
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    HAS_SPOTIFY = True
except Exception:
    HAS_SPOTIFY = False

# --- Global singletons (created lazily) ---
_PROCESSOR = None
_MODEL = None
_DEVICE = None

# Helpful runtime flags to avoid slow/buggy code paths on CPU
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYTORCH_SDP_BACKEND", "math")   # avoid SDPA path on CPU

def _lazy_import_torch_and_tf():
    """Import torch & transformers only when needed, not at Django import time."""
    # Import inside function to avoid slow startup
    import torch  # noqa: F401
    from transformers import AutoProcessor, MusicgenForConditionalGeneration, pipeline  # noqa: F401
    return torch, AutoProcessor, MusicgenForConditionalGeneration, pipeline


def _get_musicgen(model_name='facebook/musicgen-small', torch_dtype=None):
    global _PROCESSOR, _MODEL, _DEVICE
    if _PROCESSOR is not None and _MODEL is not None:
        return _PROCESSOR, _MODEL, _DEVICE

    torch, AutoProcessor, MusicgenForConditionalGeneration, _ = _lazy_import_torch_and_tf()

    dtype = None
    if torch_dtype:
        s = str(torch_dtype).lower()
        if s == 'float16':
            dtype = torch.float16
        elif s == 'bfloat16':
            dtype = torch.bfloat16
        elif s == 'float32':
            dtype = torch.float32

    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    _PROCESSOR = AutoProcessor.from_pretrained(model_name)
    _MODEL = MusicgenForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return _PROCESSOR, _MODEL, _DEVICE


def analyze_prompt(prompt: str, do_spotify: bool = True):
    """CPU-friendly analysis: emotions + sentiment + optional Spotify."""
    # Import pipeline lazily
    _, _, _, pipeline = _lazy_import_torch_and_tf()

    # Use small, reliable text models to avoid meta/SDPA issues
    emotion_pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=-1,
        model_kwargs={'attn_implementation': 'eager'},
    )
    emo_scores = emotion_pipe(prompt)[0]
    emo_scores_sorted = sorted(emo_scores, key=lambda x: x["score"], reverse=True)

    mood_map = {
        "joy": "joy",
        "neutral": "relaxed",
        "anger": "angry",
        "sadness": "sad",
        "disgust": "melancholic",
        "fear": "melancholic",
        "surprise": "energetic",
    }

    top_moods = []
    for item in emo_scores_sorted[:3]:
        mapped = mood_map.get(item["label"].lower(), item["label"].lower())
        top_moods.append((mapped, float(item["score"])))

    multi_moods = []
    for item in emo_scores_sorted:
        mapped = mood_map.get(item["label"].lower(), item["label"].lower())
        if float(item["score"]) >= 0.30:
            multi_moods.append((mapped, float(item["score"])))
    if not multi_moods and emo_scores_sorted:
        multi_moods = [(mood_map.get(emo_scores_sorted[0]["label"].lower(),
                                     emo_scores_sorted[0]["label"].lower()),
                        float(emo_scores_sorted[0]["score"]))]

    sent_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
        model_kwargs={'attn_implementation': 'eager'},
    )
    sres = sent_pipe(prompt)[0]
    sentiment_label = sres.get("label", "unknown").lower()
    sentiment_score = float(sres.get("score", 0.0))

    tracks = []
    if do_spotify and HAS_SPOTIFY:
        cid = os.environ.get('SPOTIPY_CLIENT_ID')
        csc = os.environ.get('SPOTIPY_CLIENT_SECRET')
        if cid and csc:
            try:
                auth = SpotifyClientCredentials(client_id=cid, client_secret=csc)
                sp = spotipy.Spotify(auth_manager=auth)
                results = sp.search(q=prompt, type='track', limit=3)
                for item in results.get('tracks', {}).get('items', []):
                    tracks.append({
                        'name': item.get('name'),
                        'artists': ', '.join(a.get('name') for a in item.get('artists', [])),
                        'preview_url': item.get('preview_url'),
                        'spotify_url': item.get('external_urls', {}).get('spotify'),
                    })
            except Exception:
                pass

    return {
        'top_moods': top_moods,
        'multi_moods': multi_moods,
        'sentiment_label': sentiment_label,
        'sentiment_score': sentiment_score,
        'spotify_tracks': tracks,
    }


def generate_audio(
    prompt: str,
    duration: int = 15,
    model_name: str = 'facebook/musicgen-small',
    temperature: float = 1.0,
    top_k=None,
    top_p=0.95,
    guidance_scale: float = 3.0,
    seed=None,
    torch_dtype=None,
):
    torch, _, _, _ = _lazy_import_torch_and_tf()
    processor, model, device = _get_musicgen(model_name, torch_dtype)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    max_new_tokens = int(max(2, duration) * 50)

    inputs = processor(text=[prompt], padding=True, return_tensors='pt').to(device)

    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'do_sample': True,
        'temperature': float(temperature),
        'guidance_scale': float(guidance_scale),
    }
    if top_k is not None:
        gen_kwargs['top_k'] = int(top_k)
    if top_p is not None:
        gen_kwargs['top_p'] = float(top_p)

    with torch.no_grad():
        audio = model.generate(**inputs, **gen_kwargs)

    audio_np = audio[0].detach().cpu().float().numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np[None, :]
    peak = float(np.max(np.abs(audio_np)))
    if peak > 0:
        audio_np = audio_np / peak

    sr = 32000
    try:
        sr = getattr(getattr(processor, 'feature_extractor', None), 'sampling_rate', 32000)
    except Exception:
        pass

    audio_to_save = audio_np.T if audio_np.shape[0] in (1, 2) else audio_np

    buf = io.BytesIO()
    sf.write(buf, audio_to_save, samplerate=sr, format='WAV')
    buf.seek(0)

    metadata = {
        'prompt': prompt,
        'duration_s': duration,
        'model_name': model_name,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'guidance_scale': guidance_scale,
        'seed': seed,
    }
    return buf, sr, metadata
