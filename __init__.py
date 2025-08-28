import os
import traceback
from functools import lru_cache
from transformers import logging, MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch

# ================================
# Configuración ONNX
# ================================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()

ES_EN_MODEL = "Xenova/opus-mt-es-en"
EN_ES_MODEL = "Xenova/opus-mt-en-es"


# ================================
# Traductor ONNX
# ================================
class ONNXTranslator:
    def __init__(self):
        self._loaded = False

    def _load(self):
        # Cargar tokenizadores y modelos
        self.tokenizer_es_en = MarianTokenizer.from_pretrained(ES_EN_MODEL)
        self.model_es_en = ORTModelForSeq2SeqLM.from_pretrained(ES_EN_MODEL)

        self.tokenizer_en_es = MarianTokenizer.from_pretrained(EN_ES_MODEL)
        self.model_en_es = ORTModelForSeq2SeqLM.from_pretrained(EN_ES_MODEL)

        self._loaded = True

    @lru_cache(maxsize=128)
    def translate(self, text: str, src: str, tgt: str) -> str:
        if not self._loaded:
            self._load()

        if src == "es" and tgt == "en":
            tok, model = self.tokenizer_es_en, self.model_es_en
        elif src == "en" and tgt == "es":
            tok, model = self.tokenizer_en_es, self.model_en_es
        else:
            return text

        inputs = tok([text], return_tensors="pt", padding=True)

        with torch.inference_mode():
            outputs = model.generate(**inputs)

        translated = tok.decode(outputs[0], skip_special_tokens=True)
        return translated


translator = ONNXTranslator()


# ================================
# Nodo: Traducción de texto CLIP
# ================================
class CLIPTextTranslateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "direction": (
                    ["ES→EN", "EN→ES"],
                    {"default": "ES→EN", "label": "Dirección"},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text: str, direction: str):
        src, tgt = ("es", "en") if direction == "ES→EN" else ("en", "es")

        try:
            if text.strip():
                translated_text = translator.translate(text, src, tgt)
                print(
                    f"[CLIP Translator] Traducción: '{text}' → '{translated_text}'"
                )
                text = translated_text
        except Exception as e:
            print(f"[Traductor] error al traducir: {e}")
            traceback.print_exc()

        cond, pooled = clip.encode_from_tokens(
            clip.tokenize(text), return_pooled=True
        )
        return ([[cond, {"pooled_output": pooled}]],)


# ================================
# Nodo: Traducción de texto Prompt
# ================================
class PromptTextTranslateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "direction": (
                    ["ES→EN", "EN→ES"],
                    {"default": "ES→EN", "label": "Dirección"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_value"
    CATEGORY = "conditioning"

    def get_value(self, prompt: str, direction: str):
        src, tgt = ("es", "en") if direction == "ES→EN" else ("en", "es")

        try:
            if prompt.strip():
                translated_text = translator.translate(prompt, src, tgt)
                print(
                    f"[Prompt Translator] Traducción: '{prompt}' → '{translated_text}'"
                )
                prompt = translated_text
        except Exception as e:
            print(f"[Traductor] error al traducir: {e}")
            traceback.print_exc()

        return (prompt,)


# ================================
# Registro de nodos
# ================================
NODE_CLASS_MAPPINGS = {
    "CLIPTextTranslateNode": CLIPTextTranslateNode,
    "PromptTextTranslateNode": PromptTextTranslateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextTranslateNode": "Traductor: CLIP Texto (EN↔ES)",
    "PromptTextTranslateNode": "Traductor: Prompt Texto (EN↔ES)",
}
