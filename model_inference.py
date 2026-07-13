"""
model_inference.py — FinDec 4-tier inference pipeline.

Tier 1a: NER          (PhoBERT / XLM-R)     — BIO tagging → entities
Tier 1b: Topic        (PhoBERT)              — 10-class classification → main_topic
Tier 1c: Event        (PhoBERT, multi-label) — 18-class → event_type + confidence
Tier 2:  Detail       (BARTpho / mT5, LoRA)  — seq2seq generation → context dict

Cách dùng (singleton):
    from model_inference import get_inference
    inference = get_inference()          # load lần đầu ~30s-60s
    events = inference.predict(articles) # list[dict] → list[FinancialEvent]
"""

from __future__ import annotations

import gc
import json
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# ─────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "model" / "findec_models"
CONSTANTS_PATH = MODEL_DIR / "constants.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Model architecture definitions
# (phải khớp chính xác với kiến trúc lúc training)
# ─────────────────────────────────────────────


class NERModel(nn.Module):
    """Token-level BIO classifier trên top của encoder (PhoBERT / XLM-R)."""

    def __init__(self, model_name: str, num_labels: int, hidden: int = 768) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        # Freeze 8 lớp đầu (giống training)
        layers: list = []
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        elif hasattr(self.encoder, "roberta"):
            layers = self.encoder.roberta.encoder.layer
        for i, layer in enumerate(layers):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False
        self.cls = nn.Linear(hidden, num_labels)

    def forward(self, ids: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=ids, attention_mask=am)
        return self.cls(out.last_hidden_state)

    def predict(self, ids: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(ids, am).argmax(dim=-1)


class TopicClassifier(nn.Module):
    """CLS-token → 10-class topic (PhoBERT head+tail)."""

    def __init__(self, model_name: str, n_classes: int = 10) -> None:
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name)
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_classes)
        )

    def forward(self, ids: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=ids, attention_mask=am)
        return self.head(out.last_hidden_state[:, 0, :])


class EventClassifier(nn.Module):
    """CLS-token → 18-class multi-label event type (PhoBERT)."""

    def __init__(self, model_name: str, n_classes: int = 18) -> None:
        super().__init__()
        from transformers import AutoModel

        self.bert = AutoModel.from_pretrained(model_name)
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes)
        )

    def forward(self, ids: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=ids, attention_mask=am)
        return self.head(out.last_hidden_state[:, 0, :])


# ─────────────────────────────────────────────
# Helper: JSON repair (cho output Detail model)
# ─────────────────────────────────────────────


def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    for q1, q2 in [
        ("\u201c", '"'), ("\u201d", '"'),
        ("\u2018", "'"), ("\u2019", "'"),
        ("\u300c", '"'), ("\u300d", '"'),
    ]:
        s = s.replace(q1, q2)
    return s


def _repair_json(raw: str) -> str:
    text = _norm_text(raw)
    for w, r in [
        ("eventtype", "event_type"),
        ("entitiesinvolved", "entities_involved"),
        ("roleinarticle", "role_in_article"),
        ("evidencetext", "evidence_text"),
    ]:
        text = re.sub(rf'"{w}"', f'"{r}"', text, flags=re.I)
    for pat, rep in [
        (r':\s*"\s*\{\s*"\s*$', ": null"),
        (r':\s*"\s*\{\s*$', ": null"),
        (r':\s*"\s*\{\s*"', ': null, "'),
        (r':\s*\{\s*"', ': {"'),
    ]:
        text = re.sub(pat, rep, text, flags=re.M)
    text = re.sub(r":{2,}", ":", text)
    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)
    ob = text.count("{") - text.count("}")
    if ob > 0:
        text = text.rstrip() + "}" * ob
    elif ob < 0:
        while text.count("{") < text.count("}") and text.endswith("}"):
            text = text[:-1]
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        text = text[s : e + 1]
    return text


def _regex_fallback(text: str, event_type: str) -> dict[str, Any]:
    """Fallback khi JSON parse fail — dùng regex trích field."""
    ctx: dict[str, Any] = {}
    for f in ["who", "what", "when", "where", "why", "how", "tense", "result"]:
        m = re.search(rf'"{f}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        ctx[f] = m.group(1).strip() if m and m.group(1).strip() not in ("null", "") else None
    ctx.setdefault("tense", "completed")
    ctx.setdefault("what", f"Sự kiện {event_type}")
    attrs: dict[str, Any] = {}
    ab = re.search(r'"attributes"\s*:\s*\{([^}]*)\}', text)
    if ab:
        for m in re.finditer(r'"(\w+)"\s*:\s*"?([^",}\\]+)"?', ab.group(1)):
            k, v = m.group(1), m.group(2).strip().rstrip(",")
            if v and v != "null":
                try:
                    attrs[k] = float(v) if "." in v else int(v)
                except ValueError:
                    attrs[k] = v
    em = re.search(r'"evidence_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    return {
        "context": ctx,
        "attributes": attrs,
        "evidence_text": em.group(1) if em else "",
    }


# ─────────────────────────────────────────────
# FinDecInference — singleton
# ─────────────────────────────────────────────


class FinDecInference:
    """Load và chạy toàn bộ 4-tier pipeline."""

    def __init__(self) -> None:
        self._loaded = False

        # Constants từ constants.json
        with open(CONSTANTS_PATH, encoding="utf-8") as f:
            C = json.load(f)
        self.entity_types: list[str] = C["entity_types"]
        self.bio_labels: list[str] = C["bio_labels"]
        self.id2bio: dict[int, str] = {i: l for i, l in enumerate(self.bio_labels)}
        self.main_topics: list[str] = C["main_topics"]
        self.event_types: list[str] = C["event_types"]
        self.event_counts: list[int] = C["event_counts"]

        # Models (lazy-loaded)
        self.ner_model: NERModel | None = None
        self.ner_tok = None
        self.topic_model: TopicClassifier | None = None
        self.topic_tok = None
        self.event_model: EventClassifier | None = None
        self.event_tok = None
        self.thresholds: list[float] = []
        self.detail_model = None
        self.detail_tok = None
        self.detail_max_in: int = 512
        self.detail_max_out: int = 768

    # ── Load models ──────────────────────────

    def load_models(self) -> None:
        """Load toàn bộ 4 models. Gọi 1 lần duy nhất."""
        if self._loaded:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"[FinDec] Loading models on {DEVICE}…")

        # --- NER ---
        ner_cfg = json.loads((MODEL_DIR / "ner" / "config.json").read_text())
        self.ner_model = NERModel(
            ner_cfg["model_name"],
            ner_cfg["num_labels"],
            ner_cfg.get("hidden_size", 768),
        ).to(DEVICE)
        ner_state = torch.load(MODEL_DIR / "ner" / "model.pt", map_location=DEVICE)
        # Remap key cũ "classifier." → "cls." nếu checkpoint dùng tên khác
        ner_state = {k.replace("classifier.", "cls."): v for k, v in ner_state.items()}
        self.ner_model.load_state_dict(ner_state)
        self.ner_model.eval()
        self.ner_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR / "ner" / "tokenizer"))
        print("[FinDec]  ✓ NER loaded")

        # --- Topic ---
        topic_cfg = json.loads((MODEL_DIR / "topic" / "config.json").read_text())
        self.topic_model = TopicClassifier(
            topic_cfg["model_name"], topic_cfg["num_classes"]
        ).to(DEVICE)
        self.topic_model.load_state_dict(
            torch.load(MODEL_DIR / "topic" / "model.pt", map_location=DEVICE)
        )
        self.topic_model.eval()
        self.topic_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR / "topic" / "tokenizer"))
        print("[FinDec]  ✓ Topic loaded")

        # --- Event ---
        evt_cfg = json.loads((MODEL_DIR / "event" / "config.json").read_text())
        self.event_model = EventClassifier(
            evt_cfg["model_name"], evt_cfg["num_classes"]
        ).to(DEVICE)
        self.event_model.load_state_dict(
            torch.load(MODEL_DIR / "event" / "model.pt", map_location=DEVICE)
        )
        self.event_model.eval()
        self.event_tok = AutoTokenizer.from_pretrained(str(MODEL_DIR / "event" / "tokenizer"))
        self.thresholds = json.loads((MODEL_DIR / "event" / "thresholds.json").read_text())
        print("[FinDec]  ✓ Event loaded")

        # --- Detail ---
        det_cfg = json.loads((MODEL_DIR / "detail" / "config.json").read_text())
        self.detail_max_in = det_cfg.get("max_in", 512)
        self.detail_max_out = det_cfg.get("max_out", 768)
        model_type = det_cfg.get("model_type", "mbart")
        if model_type in ("t5", "mt5"):
            from transformers import T5Tokenizer
            self.detail_tok = T5Tokenizer.from_pretrained(
                str(MODEL_DIR / "detail" / "tokenizer")
            )
        else:
            self.detail_tok = AutoTokenizer.from_pretrained(
                str(MODEL_DIR / "detail" / "tokenizer")
            )
        self.detail_model = AutoModelForSeq2SeqLM.from_pretrained(
            str(MODEL_DIR / "detail" / "model")
        ).to(DEVICE)
        self.detail_model.eval()
        print("[FinDec]  ✓ Detail loaded")

        self._loaded = True
        print(f"[FinDec] All 4 models ready on {DEVICE}")

    # ── Tier 1a: NER ─────────────────────────

    def _ner_extract_entities(self, text: str, max_len: int = 256) -> list[dict]:
        """Extract entities từ 1 chunk text."""
        enc = self.ner_tok(
            text,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_offsets_mapping=True,
        )
        offsets = enc.offset_mapping if hasattr(enc, "offset_mapping") else enc["offset_mapping"]
        ids = torch.tensor([enc["input_ids"][:max_len]], device=DEVICE)
        am = torch.tensor([enc["attention_mask"][:max_len]], device=DEVICE)
        preds = self.ner_model.predict(ids, am)[0].tolist()

        spans: list[dict] = []
        cur: dict | None = None
        for pid, (cs, ce) in zip(preds, offsets):
            label = self.id2bio.get(pid, "O")
            if label.startswith("B-"):
                if cur:
                    spans.append(cur)
                cur = {"start": cs, "end": ce, "type": label[2:]}
            elif label.startswith("I-") and cur and label[2:] == cur["type"]:
                cur["end"] = ce
            else:
                if cur:
                    spans.append(cur)
                    cur = None
        if cur:
            spans.append(cur)

        entities: list[dict] = []
        seen: set[str] = set()
        for sp in spans:
            name = text[sp["start"] : sp["end"]].strip()
            if name and len(name) >= 3 and name not in seen:
                seen.add(name)
                entities.append(
                    {
                        "name": name,
                        "type": sp["type"],
                        "role_in_article": "subject" if len(entities) < 2 else "mentioned",
                    }
                )
        return entities

    def _ner_extract_long(self, text: str, max_len: int = 256, overlap: int = 50) -> list[dict]:
        """Sliding window NER cho văn bản dài."""
        enc = self.ner_tok(text, truncation=True, max_length=max_len, return_offsets_mapping=False)
        if len(enc["input_ids"]) < max_len - 10:
            return self._ner_extract_entities(text, max_len)

        chunk_chars = int(max_len * 1.2)
        overlap_chars = int(overlap * 1.2)
        all_entities: list[dict] = []
        seen: set[tuple] = set()
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk = text[start:end]
            for ent in self._ner_extract_entities(chunk, max_len):
                key = (ent["name"].lower().strip(), ent["type"])
                if key not in seen:
                    seen.add(key)
                    all_entities.append(ent)
            if end == len(text):
                break
            start = end - overlap_chars

        for i, ent in enumerate(all_entities):
            ent["role_in_article"] = "subject" if i < 2 else "mentioned"
        return all_entities

    # ── Tier 1b: Topic ───────────────────────

    def _classify_topic(self, text: str, title: str, max_len: int = 256) -> int:
        """Head+tail sampling → topic id."""
        if len(text) <= 1500:
            inp = title + ". " + text[:1500]
        else:
            inp = title + ". " + text[:1000] + " ... " + text[-500:]
        enc = self.topic_tok(inp, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = self.topic_model(
                enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
            )
            return int(logits.argmax(dim=-1).item())

    # ── Tier 1c: Event ───────────────────────

    def _classify_events(
        self, text: str, title: str, max_len: int = 256, max_events: int = 5
    ) -> list[tuple[str, float]]:
        """Per-chunk sliding window → union of detected events."""
        chunk_chars = int(max_len * 1.2)
        overlap_chars = int(max_len * 0.3)
        all_probs: list[np.ndarray] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk = text[start:end]
            inp = title + ". " + chunk[:1500]
            enc = self.event_tok(inp, truncation=True, max_length=max_len, return_tensors="pt")
            with torch.no_grad():
                probs = torch.sigmoid(
                    self.event_model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
                )[0].cpu().numpy()
            all_probs.append(probs)
            if end == len(text):
                break
            start = end - overlap_chars

        max_probs: np.ndarray = np.max(all_probs, axis=0) if all_probs else np.zeros(18)
        detected: list[tuple[str, float]] = []
        for i, prob in enumerate(max_probs):
            if self.event_counts[i] == 0 or self.event_types[i] == "other":
                continue
            if prob >= self.thresholds[i]:
                detected.append((self.event_types[i], float(prob)))
            if len(detected) >= max_events:
                break
        return detected

    # ── Tier 2: Detail ───────────────────────

    def _detail_extract(
        self, event_type: str, text: str, title: str
    ) -> dict[str, Any]:
        """Seq2seq generation → context + attributes + evidence_text."""
        inp = f"event: {event_type}\ntext: {title}. {text[:3000]}"
        enc = self.detail_tok(
            inp,
            truncation=True,
            max_length=self.detail_max_in,
            return_tensors="pt",
        )
        with torch.no_grad():
            gen = self.detail_model.generate(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                max_length=self.detail_max_out,
                num_beams=2,
                early_stopping=True,
            )
        raw = self.detail_tok.decode(gen[0], skip_special_tokens=True)

        for parse_fn in [
            lambda x: json.loads(x),
            lambda x: json.loads(_repair_json(x)),
            lambda x: _regex_fallback(x, event_type),
        ]:
            try:
                result = parse_fn(raw)
                if isinstance(result, dict) and "context" in result:
                    return result
            except Exception:  # noqa: BLE001
                pass
        return _regex_fallback(raw, event_type)

    def _detail_extract_long(
        self, event_type: str, text: str, title: str, max_chars: int = 1500, overlap: int = 200
    ) -> dict[str, Any]:
        """Chunking wrapper cho văn bản dài."""
        if len(text) <= max_chars:
            return self._detail_extract(event_type, text, title)

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap

        chunk_results = [self._detail_extract(event_type, chunk, title) for chunk in chunks]
        if not chunk_results:
            return {"context": {}, "attributes": {}, "evidence_text": ""}

        # Merge: ưu tiên chunk có evidence_text dài nhất
        sorted_results = sorted(
            chunk_results, key=lambda r: len(r.get("evidence_text", "")), reverse=True
        )
        merged_ctx: dict[str, Any] = {}
        for field in ["who", "what", "when", "where", "why", "how", "tense", "result"]:
            for r in sorted_results:
                val = r.get("context", {}).get(field)
                if val and val != "null":
                    merged_ctx[field] = val
                    break
            if field not in merged_ctx:
                merged_ctx[field] = None
        merged_ctx.setdefault("tense", "completed")

        merged_attrs: dict[str, Any] = {}
        for r in sorted_results:
            for k, v in r.get("attributes", {}).items():
                if k not in merged_attrs and v is not None:
                    merged_attrs[k] = v

        return {
            "context": merged_ctx,
            "attributes": merged_attrs,
            "evidence_text": sorted_results[0].get("evidence_text", ""),
        }

    # ── Public API ───────────────────────────

    def predict(self, articles: list[dict]) -> list[dict]:
        """
        Chạy toàn bộ 4-tier pipeline trên danh sách articles.

        Input:  list[dict] — mỗi dict cần có:
                  title, content (optional: article_id, summary, url, published_date)

        Output: list[dict] — FinancialEvent-compatible, sẵn sàng trả về API:
                  id, main_topic, event_type, title, entities_involved,
                  context, attributes, evidence_text, confidence
        """
        if not self._loaded:
            self.load_models()

        total = len(articles)
        print(f"[FinDec] Starting inference on {total} article(s) (device={DEVICE})…")
        events_out: list[dict] = []

        for idx, art in enumerate(articles, 1):
            article_id = art.get("article_id", str(uuid.uuid4())[:8])
            title = art.get("title", "")
            content = art.get("content", "")
            summary = art.get("summary", "")

            print(f"[FinDec] [{idx}/{total}] Processing: {title[:60]}…")

            # Dùng content nếu có, fallback sang summary
            text = content if len(content) >= 50 else (summary or title)
            if not text:
                print(f"[FinDec] [{idx}/{total}] Skipped — no usable text.")
                continue

            try:
                # Tier 1a: NER
                print(f"[FinDec] [{idx}/{total}]   → Tier 1a: NER…")
                entities = self._ner_extract_long(text)
                print(f"[FinDec] [{idx}/{total}]      Found {len(entities)} entities.")

                # Tier 1b: Topic
                print(f"[FinDec] [{idx}/{total}]   → Tier 1b: Topic classification…")
                topic_id = self._classify_topic(text, title)
                main_topic = self.main_topics[topic_id] if isinstance(topic_id, int) else topic_id
                print(f"[FinDec] [{idx}/{total}]      Topic: {main_topic}")

                # Tier 1c: Event (multi-label)
                print(f"[FinDec] [{idx}/{total}]   → Tier 1c: Event classification…")
                detected = self._classify_events(text, title)
                print(f"[FinDec] [{idx}/{total}]      Detected events: {[e for e,_ in detected]}")

                if not detected:
                    print(f"[FinDec] [{idx}/{total}]      No events detected — skipping article.")
                    continue

                # Tier 2: Detail cho mỗi event type được detect
                for event_type, confidence in detected:
                    print(f"[FinDec] [{idx}/{total}]   → Tier 2: Detail extraction for '{event_type}'…")
                    detail = self._detail_extract_long(event_type, text, title)
                    print(f"[FinDec] [{idx}/{total}]      Done. evidence_text length={len(detail.get('evidence_text',''))}")

                    ctx_str = str(detail.get("context", {})) + str(detail.get("attributes", {}))
                    involved = [e["name"] for e in entities if e["name"].lower() in ctx_str.lower()]
                    if not involved:
                        involved = [e["name"] for e in entities[:3]]

                    # Sanitize: đảm bảo context/attributes luôn là dict (không phải null)
                    safe_ctx = detail.get("context") or {}
                    safe_attrs = detail.get("attributes") or {}
                    safe_evidence = detail.get("evidence_text") or ""
                    if not isinstance(safe_ctx, dict):
                        safe_ctx = {}
                    if not isinstance(safe_attrs, dict):
                        safe_attrs = {}

                    events_out.append(
                        {
                            "id": f"evt-{article_id}-{event_type[:8]}",
                            "main_topic": main_topic,
                            "event_type": event_type,
                            "title": title,
                            "entities_involved": involved,
                            "context": safe_ctx,
                            "attributes": safe_attrs,
                            "evidence_text": safe_evidence or (title + ". " + summary)[:300],
                            "confidence": round(confidence, 3),
                        }
                    )

            except Exception as exc:  # noqa: BLE001
                # Lỗi 1 article không làm crash toàn pipeline
                print(f"[FinDec] [{idx}/{total}] ERROR on article {article_id}: {exc}")
                import traceback; traceback.print_exc()
                continue

        print(f"[FinDec] Inference complete. Total events extracted: {len(events_out)}")
        for i, ev in enumerate(events_out, 1):
            print(f"[FinDec]   Event {i}: [{ev['event_type']}] {ev['title'][:60]}")
            print(f"[FinDec]     topic     : {ev['main_topic']}")
            print(f"[FinDec]     entities  : {ev['entities_involved'][:3]}")
            print(f"[FinDec]     confidence: {ev['confidence']}")
            print(f"[FinDec]     context   : {ev['context']}")
            print(f"[FinDec]     evidence  : {ev['evidence_text'][:120]}…")
        return events_out

    def unload(self) -> None:
        """Giải phóng VRAM/RAM nếu cần."""
        self.ner_model = self.topic_model = self.event_model = self.detail_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False


# ─────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────

_instance: FinDecInference | None = None


def get_inference() -> FinDecInference:
    """Trả về singleton FinDecInference (lazy load)."""
    global _instance  # noqa: PLW0603
    if _instance is None:
        _instance = FinDecInference()
    return _instance
