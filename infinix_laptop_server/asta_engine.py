#!/usr/bin/env python3
"""
=============================================================================
ASTA ENGINE (Adaptive Speech-To-Action Validation & Repair Framework)
Based on arXiv:2512.12769v2 ("Adaptive Edge-Cloud Inference for Speech-to-Action Systems")

Modules:
1. MetricsCollector: System metrics (CPU workload %, RAM, Latency).
2. ASTACommandValidator: 3-Layer Command Validation & Automatic History-Aware Repair.
3. ASTARouter: Adaptive Edge/Cloud decision router.
=============================================================================
"""

import re
import time
import psutil

# Supported IoT Devices & Actions
SUPPORTED_ACTIONS = {
    "turn_on": ["turn on", "turn_on", "switch on", "start", "chalao", "on karo", "on", "activate", "open"],
    "turn_off": ["turn off", "turn_off", "switch off", "stop", "band karo", "off karo", "off", "deactivate", "close"],
    "toggle": ["toggle", "badlo", "change"],
    "status": ["status", "check", "batao", "state"]
}

SUPPORTED_DEVICES = {
    "light": ["light", "lights", "bulb", "lamp", "roshni"],
    "fan": ["fan", "fans", "pankha"],
    "ac": ["ac", "air conditioner", "cooler"],
    "switch": ["switch", "plug", "socket"],
    "door": ["door", "gate", "lock"]
}

# Phonetic & Acoustic Normalization Dictionary
PHONETIC_REPLACEMENTS = [
    (r"\bof\b", "off"),
    (r"\bto\b", "two"),
    (r"\btoo\b", "two"),
    (r"\bfor\b", "four"),
    (r"\bwon\b", "one"),
    (r"\bfun\b", "fan"),
    (r"\blite\b", "light"),
    (r"\bflight\b", "light"),
    (r"\bnight\b", "light"),
    (r"\bchalao\b", "turn on"),
    (r"\bband karo\b", "turn off"),
    (r"\bon karo\b", "turn on"),
    (r"\boff karo\b", "turn off"),
]

class MetricsCollector:
    """Monitors system indicators (CPU load %, RAM %, ASR latency) as per ASTA paper."""
    @staticmethod
    def get_metrics(asr_latency_ms=0, audio_dur_ms=0):
        cpu_workload = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        return {
            "cpu_workload_pct": round(cpu_workload, 1),
            "ram_used_pct": round(ram.percent, 1),
            "asr_latency_ms": asr_latency_ms,
            "audio_duration_ms": audio_dur_ms,
            "status": "NORMAL" if cpu_workload < 80.0 else "HIGH_LOAD"
        }

class CommandHistoryTable:
    """Tracks executed commands for ASTA history-aware repair."""
    def __init__(self, max_history=20):
        self.history = []
        self.max_history = max_history

    def add(self, action, device, index):
        self.history.insert(0, {"action": action, "device": device, "index": index, "timestamp": time.time()})
        if len(self.history) > self.max_history:
            self.history.pop()

    def get_most_frequent_index(self, device):
        indices = [item["index"] for item in self.history if item["device"] == device]
        if indices:
            return max(set(indices), key=indices.count)
        return 1  # Default fallback index 1

history_table = CommandHistoryTable()

class ASTACommandValidator:
    """3-Layer Command Validation and Automatic Repair Mechanism."""

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        for pattern, repl in PHONETIC_REPLACEMENTS:
            text = re.sub(pattern, repl, text)
        return text

    @classmethod
    def validate_and_repair(cls, raw_text: str):
        normalized = cls.normalize_text(raw_text)
        was_repaired = False
        repair_reasons = []

        if raw_text.lower().strip() != normalized:
            was_repaired = True
            repair_reasons.append("Phonetic/Acoustic STT correction applied")

        # 1. Action Layer
        detected_action = None
        for action, triggers in SUPPORTED_ACTIONS.items():
            for trig in triggers:
                if trig in normalized:
                    detected_action = action
                    break
            if detected_action:
                break

        # 2. Device Layer
        detected_device = None
        for device, synonyms in SUPPORTED_DEVICES.items():
            for syn in synonyms:
                if syn in normalized:
                    detected_device = device
                    break
            if detected_device:
                break

        # 3. Index Layer
        detected_index = None
        match = re.search(r"\b(\d+)\b", normalized)
        if match:
            detected_index = int(match.group(1))
        else:
            word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
            for w, num in word_to_num.items():
                if w in normalized:
                    detected_index = num
                    break

        # Automatic Command Repair Phase
        if detected_action and detected_device:
            if detected_index is None:
                # History-aware index repair (ASTA Paper Section IV-D)
                detected_index = history_table.get_most_frequent_index(detected_device)
                was_repaired = True
                repair_reasons.append(f"Inferred missing index '{detected_index}' from command history")

            repaired_command = f"{detected_action.upper()}_{detected_device.upper()}_{detected_index}"
            history_table.add(detected_action, detected_device, detected_index)

            return {
                "valid": True,
                "original_text": raw_text,
                "normalized_text": normalized,
                "repaired_command": repaired_command,
                "action": detected_action,
                "device": detected_device,
                "index": detected_index,
                "was_repaired": was_repaired,
                "repair_reasons": repair_reasons,
                "status": "EXECUTABLE"
            }
        else:
            return {
                "valid": False,
                "original_text": raw_text,
                "normalized_text": normalized,
                "repaired_command": None,
                "action": detected_action,
                "device": detected_device,
                "index": detected_index,
                "was_repaired": False,
                "repair_reasons": ["Action or Device missing in voice input"],
                "status": "UNRECOGNIZED_COMMAND"
            }

class ASTARouter:
    """Dynamic Metric-Aware Router between Local & Cloud Inference."""
    @staticmethod
    def route(metrics):
        if metrics["cpu_workload_pct"] > 80.0 or metrics["asr_latency_ms"] > 1500:
            return "CLOUD_FALLBACK"
        return "LOCAL_EDGE"
