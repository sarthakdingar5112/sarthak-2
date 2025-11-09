from django.contrib import admin
from .models import Prompt, Analysis, Generation

@admin.register(Prompt)
class PromptAdmin(admin.ModelAdmin):
    list_display = ('id', 'short_text', 'created_at')
    search_fields = ('text',)
    ordering = ('-created_at',)

    def short_text(self, obj):
        return (obj.text or '')[:80]

@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'prompt', 'sentiment_label', 'created_at')
    ordering = ('-created_at',)

@admin.register(Generation)
class GenerationAdmin(admin.ModelAdmin):
    list_display = ('id', 'prompt', 'duration_s', 'model_name', 'created_at')
    ordering = ('-created_at',)
