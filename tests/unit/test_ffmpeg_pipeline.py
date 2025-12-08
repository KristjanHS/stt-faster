from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.preprocess.config import PreprocessConfig
from backend.preprocess.steps.ffmpeg_pipeline import run_ffmpeg_pipeline


def test_arnndn_filter_included(tmp_path: Path, default_rnnoise_mix: float) -> None:
    """When an rnnoise model is provided, the ffmpeg filter graph must include arnndn with the model and mix."""
    input_file = tmp_path / "in.wav"
    output_file = tmp_path / "out.wav"
    input_file.write_bytes(b"\x00\x01")

    rnnoise_model = str(tmp_path / "models" / "test_model.rnnn")
    # ensure the model file exists locally for unit test (avoid network)
    model_path = tmp_path / "models"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "test_model.rnnn").write_bytes(b"dummy-model")
    rnnoise_mix = default_rnnoise_mix

    # Mock ffmpeg-python calls
    with patch("backend.preprocess.steps.ffmpeg_pipeline.ffmpeg") as mock_ffmpeg:
        # Setup mock chain: input -> filter -> output -> run
        mock_input = MagicMock()
        mock_filtered = MagicMock()
        mock_output = MagicMock()
        mock_ffmpeg.input.return_value = mock_input
        mock_input.filter.return_value = mock_filtered
        mock_ffmpeg.filter.return_value = mock_filtered
        mock_ffmpeg.output.return_value = mock_output
        mock_ffmpeg.run.return_value = None

        metrics = run_ffmpeg_pipeline(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            rnnoise_mix=rnnoise_mix,
            rnnoise_model=rnnoise_model,
        )

        # Verify ffmpeg.input was called with correct path
        mock_ffmpeg.input.assert_called_once_with(str(input_file))

        # Verify output was called with af parameter containing filter graph
        output_call = mock_ffmpeg.output.call_args
        assert "af" in output_call[1]
        filter_graph = output_call[1]["af"]

        # The arnndn filter must reference the model path and mix value
        assert f"arnndn=m={rnnoise_model}:mix={rnnoise_mix}" in filter_graph

        # Verify output was called with correct parameters
        mock_ffmpeg.output.assert_called_once()
        output_call = mock_ffmpeg.output.call_args
        assert output_call[1]["ac"] == 1
        assert output_call[1]["ar"] == 16000
        assert output_call[1]["sample_fmt"] == "s16"
        assert "af" in output_call[1]

        # Verify run was called
        mock_ffmpeg.run.assert_called_once()

        # Basic sanity on returned metrics
        assert metrics.name == "ffmpeg_pipeline"
        assert metrics.backend == "ffmpeg-python"


def test_filter_graph_includes_all_filters(tmp_path: Path, default_rnnoise_mix: float) -> None:
    """Test that filter graph includes highpass, aresample, and loudnorm."""
    input_file = tmp_path / "in.wav"
    output_file = tmp_path / "out.wav"
    input_file.write_bytes(b"\x00\x01")

    with patch("backend.preprocess.steps.ffmpeg_pipeline.ffmpeg") as mock_ffmpeg:
        mock_input = MagicMock()
        mock_ffmpeg.input.return_value = mock_input
        mock_ffmpeg.output.return_value = MagicMock()
        mock_ffmpeg.run.return_value = None

        run_ffmpeg_pipeline(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            rnnoise_mix=default_rnnoise_mix,
        )

        # Get the filter graph from the output call
        output_call = mock_ffmpeg.output.call_args
        filter_graph = output_call[1]["af"]

        # Verify all expected filters are present
        assert "highpass=f=80:poles=2" in filter_graph
        assert "aresample=resampler=soxr:osr=16000" in filter_graph
        assert "loudnorm=I=-20.0:TP=-2.0:LRA=7.0" in filter_graph


def test_loudnorm_preset_selection(tmp_path: Path, default_rnnoise_mix: float) -> None:
    """Test that different loudnorm presets use correct values."""
    input_file = tmp_path / "in.wav"
    output_file = tmp_path / "out.wav"
    input_file.write_bytes(b"\x00\x01")

    with patch("backend.preprocess.steps.ffmpeg_pipeline.ffmpeg") as mock_ffmpeg:
        mock_input = MagicMock()
        mock_filtered = MagicMock()
        mock_ffmpeg.input.return_value = mock_input
        mock_ffmpeg.filter.return_value = mock_filtered
        mock_ffmpeg.output.return_value = MagicMock()
        mock_ffmpeg.run.return_value = None

        run_ffmpeg_pipeline(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            loudnorm_preset="boost-quiet-voices",
            rnnoise_mix=default_rnnoise_mix,
        )

        # Get the filter graph from the output call
        output_call = mock_ffmpeg.output.call_args
        filter_graph = output_call[1]["af"]

        # Verify boost-quiet-voices preset values
        assert "loudnorm=I=-18.0:TP=-2.0:LRA=7.0" in filter_graph


def test_default_rnnoise_model_uses_sh_rnnn(tmp_path: Path, default_rnnoise_mix: float) -> None:
    """Test that the default PreprocessConfig uses sh.rnnn model path."""
    input_file = tmp_path / "in.wav"
    output_file = tmp_path / "out.wav"
    input_file.write_bytes(b"\x00\x01")

    # Create the default model file to avoid download
    default_model_path = tmp_path / "models" / "sh.rnnn"
    default_model_path.parent.mkdir(parents=True, exist_ok=True)
    default_model_path.write_bytes(b"dummy-sh-model")

    # Use default config which should have models/sh.rnnn
    cfg = PreprocessConfig()
    assert cfg.rnnoise_model == "models/sh.rnnn"

    # Adjust the model path to use tmp_path for the test
    test_model_path = str(default_model_path)

    with patch("backend.preprocess.steps.ffmpeg_pipeline.ffmpeg") as mock_ffmpeg:
        mock_input = MagicMock()
        mock_ffmpeg.input.return_value = mock_input
        mock_ffmpeg.output.return_value = MagicMock()
        mock_ffmpeg.run.return_value = None

        run_ffmpeg_pipeline(
            input_path=input_file,
            output_path=output_file,
            target_sample_rate=16000,
            target_channels=1,
            rnnoise_mix=default_rnnoise_mix,
            rnnoise_model=test_model_path,
        )

        # Get the filter graph from the output call
        output_call = mock_ffmpeg.output.call_args
        filter_graph = output_call[1]["af"]

        # Verify the filter graph includes arnndn with the sh.rnnn model
        assert f"arnndn=m={test_model_path}:mix={default_rnnoise_mix}" in filter_graph
