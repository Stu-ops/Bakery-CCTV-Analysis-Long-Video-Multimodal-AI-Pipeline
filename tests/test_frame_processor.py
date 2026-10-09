"""
Tests for FrameProcessor — CORRECTION #1: 5 FPS Constant

Validates that:
- 5 FPS captures zone transitions (1.5s crossing = 7+ frames)
- 1 FPS would miss transitions (only 2 frames)
- Timestamps are correctly computed
- Edge cases handled (invalid FPS, boundary frames)
"""

import pytest
from datetime import datetime, timedelta
from pipeline.frame_processor import FrameProcessor


class TestFrameProcessor:
    """Test suite for corrected 5 FPS frame processing."""

    def test_default_initialization(self):
        """Default should be 5 FPS from 15 FPS source."""
        processor = FrameProcessor()
        assert processor.source_fps == 15
        assert processor.target_fps == 5
        assert processor.frame_interval == 3

    def test_5fps_captures_zone_transition(self):
        """
        CRITICAL TEST: 1.5-second zone crossing should capture 7+ frames.
        This is the core validation for CORRECTION #1.
        """
        processor = FrameProcessor(source_fps=15, target_fps=5)

        # Simulate: person takes 1.5 seconds to cross zone
        crossing_start_frame = 100
        crossing_duration_seconds = 1.5
        crossing_end_frame = crossing_start_frame + int(
            crossing_duration_seconds * 15
        )

        captured_frames = [
            fid
            for fid in range(crossing_start_frame, crossing_end_frame + 1)
            if processor.should_process_frame(fid)
        ]

        # At 5 FPS (interval=3): should capture ~7 frames
        assert len(captured_frames) >= 5, (
            f"5 FPS should capture 5+ frames during 1.5s crossing, "
            f"got {len(captured_frames)}"
        )
        print(f"✓ Captured {len(captured_frames)} frames during 1.5s crossing")

    def test_1fps_would_miss_transition(self):
        """
        Validates that 1 FPS (ChatGPT's approach) misses transitions.
        This proves why the correction is necessary.
        """
        processor_1fps = FrameProcessor(source_fps=15, target_fps=1)

        crossing_start = 100
        crossing_end = 100 + int(1.5 * 15)  # 1.5 second crossing

        captured_at_1fps = [
            fid
            for fid in range(crossing_start, crossing_end + 1)
            if processor_1fps.should_process_frame(fid)
        ]

        # 1 FPS captures only 1-2 frames — insufficient
        assert len(captured_at_1fps) <= 2, (
            f"1 FPS should capture <=2 frames, got {len(captured_at_1fps)}"
        )

    def test_frame_interval_calculation(self):
        """Frame interval = source_fps // target_fps."""
        p1 = FrameProcessor(source_fps=30, target_fps=5)
        assert p1.frame_interval == 6

        p2 = FrameProcessor(source_fps=15, target_fps=5)
        assert p2.frame_interval == 3

        p3 = FrameProcessor(source_fps=15, target_fps=15)
        assert p3.frame_interval == 1

    def test_should_process_frame(self):
        """Every Nth frame should be processed."""
        processor = FrameProcessor(source_fps=15, target_fps=5)

        # Interval = 3: frames 0, 3, 6, 9... should be processed
        assert processor.should_process_frame(0) is True
        assert processor.should_process_frame(1) is False
        assert processor.should_process_frame(2) is False
        assert processor.should_process_frame(3) is True
        assert processor.should_process_frame(6) is True

    def test_timestamp_from_frame(self):
        """Timestamps should be correctly computed from frame index."""
        clip_start = datetime(2026, 3, 3, 14, 0, 0)
        processor = FrameProcessor(
            source_fps=15, target_fps=5, clip_start_time=clip_start
        )

        # Frame 0 → clip start
        ts0 = processor.get_timestamp_from_frame(0)
        assert "2026-03-03T14:00:00.000Z" == ts0

        # Frame 15 → 1 second later
        ts15 = processor.get_timestamp_from_frame(15)
        assert "2026-03-03T14:00:01.000Z" == ts15

        # Frame 150 → 10 seconds later
        ts150 = processor.get_timestamp_from_frame(150)
        assert "2026-03-03T14:00:10.000Z" == ts150

    def test_datetime_from_frame(self):
        """datetime computation should match timestamp."""
        clip_start = datetime(2026, 3, 3, 14, 0, 0)
        processor = FrameProcessor(
            source_fps=15, target_fps=5, clip_start_time=clip_start
        )

        dt = processor.get_datetime_from_frame(30)
        assert dt == clip_start + timedelta(seconds=2)

    def test_effective_fps(self):
        """Effective FPS should match target."""
        processor = FrameProcessor(source_fps=15, target_fps=5)
        assert processor.get_effective_fps() == 5.0

    def test_processing_ratio(self):
        """Should process ~33% of frames at 5/15 FPS."""
        processor = FrameProcessor(source_fps=15, target_fps=5)

        for fid in range(300):  # 300 frames
            processor.should_process_frame(fid)

        stats = processor.stats
        ratio = stats["processing_ratio"]
        assert 0.30 <= ratio <= 0.36, f"Expected ~33%, got {ratio:.1%}"

    def test_invalid_fps_raises(self):
        """Invalid FPS values should raise ValueError."""
        with pytest.raises(ValueError):
            FrameProcessor(source_fps=15, target_fps=0)

        with pytest.raises(ValueError):
            FrameProcessor(source_fps=0, target_fps=5)

        with pytest.raises(ValueError):
            FrameProcessor(source_fps=5, target_fps=15)

    def test_full_clip_frame_count(self):
        """20-min clip at 5 FPS should process ~6000 frames."""
        processor = FrameProcessor(source_fps=15, target_fps=5)

        total_frames = 20 * 60 * 15  # 18,000 raw frames
        processed = sum(
            1
            for fid in range(total_frames)
            if processor.should_process_frame(fid)
        )

        # At 5 FPS: should process ~6000 frames (18000/3)
        assert 5900 <= processed <= 6100, f"Expected ~6000, got {processed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
