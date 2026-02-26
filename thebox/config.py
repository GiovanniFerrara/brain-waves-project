"""TheBox configuration — dataclass-based config with sensible defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TheBoxConfig:
    # BLE
    device_name: str = "Muse-31A9"
    scan_timeout: float = 10.0
    connect_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 2.0

    # EEG
    eeg_buffer_seconds: float = 10.0
    sample_rate: int = 256

    # Pipeline
    process_interval: float = 0.05  # 50ms = 20 Hz processing loop

    # Event detection
    blink_threshold: float = 200.0      # µV peak-to-peak on AF7+AF8
    blink_window: float = 0.2           # 200ms detection window
    blink_debounce: float = 0.3         # 300ms debounce
    clench_threshold: float = 30.0      # RMS threshold for jaw clench
    clench_window: float = 0.5          # 500ms detection window
    clench_debounce: float = 0.5        # 500ms debounce
    alpha_burst_ratio: float = 1.5      # alpha power vs baseline ratio
    alpha_baseline_seconds: float = 10.0  # rolling baseline window

    # Sound
    audio_sample_rate: int = 44100
    audio_block_size: int = 2205        # ~50ms at 44100 Hz
    audio_channels: int = 1
    base_frequency_range: tuple[float, float] = (110.0, 880.0)
    blink_decay: float = 0.2            # seconds
    clench_decay: float = 0.3           # seconds

    # Output
    master_volume: float = 0.5
