# ComfyUI-traductor-offline/__init__.py
# Nodo traductor (EN↔ES) optimizado: precarga opcional, GPU si disponible, batch support,
# logging, saneamiento, chunking para textos largos y splitting lógico por ., , or ).
import os
import traceback
import logging
import re
from functools import lru_cache
from typing import List, Union, Optional

import torch
from transformers import logging as hf_logging, MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

# --- Configuración inicial ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
hf_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
)
logger = logging.getLogger("ONNXTranslator")

ES_EN_MODEL = "Xenova/opus-mt-es-en"
EN_ES_MODEL = "Xenova/opus-mt-en-es"

# Máximo tokens por chunk (sensible por debajo del límite típico ~512)
DEFAULT_MAX_TOKENS = 480

# Abreviaturas comunes para evitar splits después de "Sr.", "etc.", etc.
COMMON_ABBREVIATIONS = {
    "sr.", "sra.", "srta.", "dr.", "dra.", "prof.", "lic.", "cap.", "av.",
    "etc.", "ej.", "e.g.", "i.e.", "mr.", "mrs.", "ms.", "jr.", "sr", "sra",
    "sra.", "u.s.a.", "u.k.", "a.m.", "p.m."
}


def _sanitize_text(t: str) -> str:
    """Sanitiza el texto: trim, colapsa saltos de línea y tabulaciones."""
    if not isinstance(t, str):
        return t
    s = t.strip()
    # Reemplaza múltiples saltos de línea por espacios
    s = " ".join(s.split())
    return s


def _likely_english(text: str, ascii_threshold: float = 0.9) -> bool:
    """
    Heurística simple: si gran parte de los chars son ASCII imprimibles,
    asumimos que probablemente ya esté en inglés (útil para omitir traducción).
    No es una detección perfecta; sólo optimiza casos obvios.
    """
    if not text:
        return True
    printable = sum(1 for c in text if 32 <= ord(c) < 127)
    return (printable / len(text)) >= ascii_threshold


def _split_text_logical(text: str) -> List[str]:
    """
    Divide el texto en segmentos lógicos, cortando preferentemente en:
      - puntos (.), comas (,) o paréntesis de cierre `)` seguidos de espacio
    Evita cortar en abreviaturas comunes, números decimales o iniciales.
    Conserva la puntuación al final de cada segmento.
    """
    text = text.strip()
    if not text:
        return []

    splits = []
    last_idx = 0

    # Buscar posibles puntos de corte: posición donde hay '.' or ',' or ')' followed by whitespace
    for m in re.finditer(r'(?<=[\.,\)])\s+', text):
        cut_pos = m.start()  # índice donde empieza el espacio siguiente al signo
        # Preceding text up to cut_pos (trimmed right)
        prev = text[:cut_pos].rstrip()

        # Extraer la "token" final (palabra + posible punto)
        token_match = re.search(r'([A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\.-]{1,20})\.*$', prev)
        token = token_match.group(1).lower() if token_match else ""

        # Evitar split si es una abreviatura conocida (ej: "Sr.", "etc.")
        if token.endswith('.') and token in COMMON_ABBREVIATIONS:
            continue

        # Evitar split si parece un número decimal al final (ej: "3.14")
        if re.search(r'\d+\.\d+$', prev):
            continue

        # Evitar split si es una inicial (ej: "J.")
        if re.search(r'\b[A-ZÁÉÍÓÚÑ]\.$', prev):
            continue

        # Evitar split si la "token" es una sigla/URL/email parcial (heurística simple)
        if re.search(r'@\w+$', prev) or re.search(r'\w+\.\w{2,4}$', prev):
            # esto evita cortar dentro de dominios o abreviaturas raras
            continue

        # Si llegamos aquí, es un split válido
        splits.append(text[last_idx:cut_pos].strip())
        last_idx = m.end()

    # Añadir resto
    remainder = text[last_idx:].strip()
    if remainder:
        splits.append(remainder)

    # Si no se detectó ningún split (lista vacía), devolver el texto entero
    if not splits:
        return [text]

    return [s for s in splits if s]


def _build_chunks_from_segments(segments: List[str], tokenizer: MarianTokenizer, max_tokens: int) -> List[str]:
    """
    Agrupa segmentos lógicos en chunks cuyo total estimado de tokens <= max_tokens.
    Si un segmento individual excede max_tokens, se divide por palabras (fallback).
    """
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for seg in segments:
        token_ids = tokenizer.encode(seg, add_special_tokens=False)
        seg_len = len(token_ids)

        if seg_len > max_tokens:
            # dividir por palabras si el segmento es demasiado largo
            words = seg.split()
            subchunk = []
            sublen = 0
            for w in words:
                w_len = len(tokenizer.encode(w, add_special_tokens=False))
                if sublen + w_len > max_tokens:
                    if subchunk:
                        # flush subchunk
                        if current_len + sublen <= max_tokens:
                            current_chunk.append(" ".join(subchunk))
                            current_len += sublen
                        else:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            current_chunk = [" ".join(subchunk)]
                            current_len = sublen
                        subchunk = []
                        sublen = 0
                    else:
                        # palabra sola mayor que max_tokens (extremo): la forzamos como chunk
                        chunks.append(w)
                        subchunk = []
                        sublen = 0
                else:
                    subchunk.append(w)
                    sublen += w_len
            if subchunk:
                if current_len + sublen <= max_tokens:
                    current_chunk.append(" ".join(subchunk))
                    current_len += sublen
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    chunks.append(" ".join(subchunk))
                    current_chunk = []
                    current_len = 0
            continue

        # si el segmento entra en el chunk actual
        if current_len + seg_len <= max_tokens:
            current_chunk.append(seg)
            current_len += seg_len
        else:
            # flush current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [seg]
            current_len = seg_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]


class ONNXTranslator:
    def __init__(self, use_gpu_if_available: bool = True, preload: bool = False, max_tokens: int = DEFAULT_MAX_TOKENS):
        """
        use_gpu_if_available: intenta usar ejecución CUDA si torch.cuda.is_available()
        preload: si True carga modelos en la inicialización (evita latencia en la 1ª traducción)
        max_tokens: tokens máximos por chunk (para textos largos)
        """
        self._loaded = False
        self._use_gpu_if_available = use_gpu_if_available
        self._providers = None  # for debugging/inspect
        self.max_tokens = int(max_tokens)
        if preload:
            try:
                self._load()
            except Exception:
                logger.exception("Error during preload; will attempt lazy load on first use.")

    def _load(self):
        if self._loaded:
            return

        # Decide providers based on torch.cuda availability and user preference
        cuda_available = torch.cuda.is_available() and self._use_gpu_if_available
        try:
            if cuda_available:
                logger.info("CUDA disponible según torch. Intentando cargar modelos con CUDAExecutionProvider.")
                self.tokenizer_es_en = MarianTokenizer.from_pretrained(ES_EN_MODEL)
                self.model_es_en = ORTModelForSeq2SeqLM.from_pretrained(
                    ES_EN_MODEL, providers=["CUDAExecutionProvider"]
                )
                self.tokenizer_en_es = MarianTokenizer.from_pretrained(EN_ES_MODEL)
                self.model_en_es = ORTModelForSeq2SeqLM.from_pretrained(
                    EN_ES_MODEL, providers=["CUDAExecutionProvider"]
                )
                self._providers = ["CUDAExecutionProvider"]
            else:
                raise RuntimeError("Forzando fallback a CPU (CUDA no disponible o deshabilitado).")
        except Exception as e_cuda:
            logger.warning(
                "No se pudo inicializar con CUDAExecutionProvider (o no está disponible). "
                "Cargando en CPU como fallback. Error: %s", e_cuda
            )
            # Fallback a CPU providers (default)
            self.tokenizer_es_en = MarianTokenizer.from_pretrained(ES_EN_MODEL)
            self.model_es_en = ORTModelForSeq2SeqLM.from_pretrained(ES_EN_MODEL)
            self.tokenizer_en_es = MarianTokenizer.from_pretrained(EN_ES_MODEL)
            self.model_en_es = ORTModelForSeq2SeqLM.from_pretrained(EN_ES_MODEL)
            self._providers = ["CPUExecutionProvider" if not cuda_available else "unknown"]

        self._loaded = True
        logger.info("Modelos cargados. Providers: %s", self._providers)

    def _translate_chunk(self, chunk: str, tok: MarianTokenizer, model: ORTModelForSeq2SeqLM) -> str:
        """
        Traduce un chunk que se asume dentro de max_tokens (no hace más splitting).
        No decorado con cache (la cache es por segmento en _translate_single).
        """
        try:
            inputs = tok([chunk], return_tensors="pt", padding=True, truncation=False)
            with torch.inference_mode():
                outputs = model.generate(**inputs)
            translated = tok.decode(outputs[0], skip_special_tokens=True)
            return _sanitize_text(translated)
        except Exception:
            logger.exception("Error en generación del modelo para chunk (fallback a truncado).")
            # fallback: truncar y reintentar
            inputs = tok([chunk], return_tensors="pt", padding=True, truncation=True, max_length=self.max_tokens)
            with torch.inference_mode():
                outputs = model.generate(**inputs)
            translated = tok.decode(outputs[0], skip_special_tokens=True)
            return _sanitize_text(translated)

    @lru_cache(maxsize=2048)
    def _translate_single(self, text: str, src: str, tgt: str) -> str:
        """
        Traduce un único string (interno, cacheable).
        Maneja textos largos mediante:
          1) split lógico (.,, ) ),
          2) agrupación en chunks según tokens,
          3) traducción por chunk y reensamblado.
        """
        text = _sanitize_text(text)
        if not text:
            return text

        # Bypass heurístico: ES->EN si ya parece inglés ASCII
        if src == "es" and tgt == "en" and _likely_english(text):
            logger.debug("Bypass de traducción ES→EN: texto parece ya ASCII/inglés: %s", text[:60])
            return text

        # Selección de tokenizer y modelo según par
        if src == "es" and tgt == "en":
            tok, model = self.tokenizer_es_en, self.model_es_en
        elif src == "en" and tgt == "es":
            tok, model = self.tokenizer_en_es, self.model_en_es
        else:
            return text

        try:
            input_ids_len = len(tok.encode(text, add_special_tokens=False))

            if input_ids_len <= self.max_tokens:
                # caso simple
                return self._translate_chunk(text, tok, model)

            # Texto largo: hacemos split lógico y build de chunks
            segments = _split_text_logical(text)
            chunks = _build_chunks_from_segments(segments, tok, self.max_tokens)

            translated_parts: List[str] = []
            for chunk in chunks:
                translated_parts.append(self._translate_chunk(chunk, tok, model))

            # Unir partes; usamos doble espacio para reducir fusiones accidentales
            joined = "  ".join([p for p in translated_parts if p])
            return _sanitize_text(joined)

        except Exception as e:
            logger.exception("Error en generación del modelo para texto largo: %s", str(e)[:200])
            return text

    def translate(self, text: Union[str, List[str]], src: str, tgt: str) -> Union[str, List[str]]:
        """
        API pública: acepta un string o lista de strings y devuelve la traducción correspondiente.
        Usa caching por item (interno) y pre-carga de modelos.
        """
        if not self._loaded:
            logger.info("Modelos no cargados todavía — cargando ahora.")
            try:
                self._load()
            except Exception as e:
                logger.exception("Fallo al cargar los modelos: %s", e)
                return text

        if isinstance(text, str):
            try:
                result = self._translate_single(text, src, tgt)
                logger.debug("Traducción: '%s' -> '%s'", text[:80], result[:80])
                return result
            except Exception:
                logger.exception("Error traduciendo string unico; devolviendo original.")
                return text
        elif isinstance(text, list):
            out: List[str] = []
            # Para listas grandes se podría implementar batch generation; por ahora usamos cache por elemento
            for t in text:
                try:
                    out.append(self._translate_single(t, src, tgt))
                except Exception:
                    logger.exception("Error traduciendo elemento de la lista; devolviendo original para ese elemento.")
                    out.append(t)
            return out
        else:
            logger.warning("Tipo no soportado para translate(): %s", type(text))
            return text


# Instancia global (preload opcional - por defecto False para evitar cargas masivas si hay muchos nodos)
translator = ONNXTranslator(preload=False)

# ------------------- NODO CLIP ---------------------
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
            if text and text.strip():
                translated_text = translator.translate(text, src, tgt)
                if isinstance(translated_text, list):
                    translated_text = " ".join(translated_text)
                logger.info("[CLIP Translator] '%s' → '%s'", text[:80], translated_text[:80])
                text = translated_text
        except Exception as e:
            logger.exception("[Traductor] error al traducir en CLIP node: %s", e)

        cond, pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]],)


# ------------------- NODO PROMPT ---------------------
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
            if prompt and prompt.strip():
                translated_text = translator.translate(prompt, src, tgt)
                if isinstance(translated_text, list):
                    translated_text = " ".join(translated_text)
                logger.info("[Prompt Translator] '%s' → '%s'", prompt[:80], translated_text[:80])
                prompt = translated_text
        except Exception as e:
            logger.exception("[Traductor] error al traducir en Prompt node: %s", e)

        return (prompt,)


# ------------------- REGISTRO ---------------------
NODE_CLASS_MAPPINGS = {
    "CLIPTextTranslateNode": CLIPTextTranslateNode,
    "PromptTextTranslateNode": PromptTextTranslateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextTranslateNode": "Traductor: CLIP Texto (EN↔ES) — optimizado",
    "PromptTextTranslateNode": "Traductor: Prompt Texto (EN↔ES) — optimizado",
}
