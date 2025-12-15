# Statistics Completeness Analysis

This document reviews all expected statistics in `check_parameter_completeness.py` and verifies whether they are actually calculated and stored in the codebase.

## Summary

**All expected statistics ARE calculated**, except for `input_bit_depth` which is intentionally not available (see details below).

## Job Level Statistics (runs table)

All job-level statistics in `RUN_STATISTICS_COLUMNS` are properly calculated:

| Statistic | Calculated? | Location |
|-----------|-------------|----------|
| `files_found` | ✅ Yes | `backend/processor.py:508` - Count of files to process |
| `succeeded` | ✅ Yes | `backend/processor.py:686` - Count from results dict |
| `failed` | ✅ Yes | `backend/processor.py:687` - Count from results dict |
| `total_processing_time` | ✅ Yes | `backend/processor.py:688` - Total time from run_start to completion |
| `total_preprocess_time` | ✅ Yes | `backend/processor.py:525,689` - Sum of all file preprocess_duration |
| `total_transcribe_time` | ✅ Yes | `backend/processor.py:526,690` - Sum of all file transcribe_duration |
| `total_audio_duration` | ✅ Yes | `backend/processor.py:527-528,691` - Sum of all file audio_duration |
| `speed_ratio` | ✅ Yes | `backend/processor.py:530-531,692` - Average of all file speed_ratio values |

**Implementation details:**
- Statistics are calculated in `TranscriptionProcessor._record_run_metadata()` 
- Aggregates are computed from individual file metrics
- All values are stored via `RunRecord` to `db.record_run()`

## File Level Statistics (file_metrics table)

All file-level statistics in `FILE_STATISTICS_COLUMNS` are properly calculated:

| Statistic | Calculated? | Source |
|-----------|-------------|--------|
| `status` | ✅ Yes | Always set (required NOT NULL field) - `backend/processor.py:711,728` |
| `audio_duration` | ✅ Yes | From `TranscriptionMetrics.audio_duration` - `backend/transcribe.py:628, processor.py:733` |
| `total_processing_time` | ✅ Yes | From `TranscriptionMetrics.total_processing_time` - `backend/transcribe.py:629, processor.py:734` |
| `transcribe_duration` | ✅ Yes | From `TranscriptionMetrics.transcribe_duration` - `backend/transcribe.py:630, processor.py:735` |
| `preprocess_duration` | ✅ Yes | From `TranscriptionMetrics.preprocess_duration` - `backend/transcribe.py:631, processor.py:736` |
| `speed_ratio` | ✅ Yes | From `TranscriptionMetrics.speed_ratio` - `backend/transcribe.py:632, processor.py:737` |
| `input_channels` | ✅ Yes | From `TranscriptionMetrics.input_channels` - `backend/transcribe.py:644, processor.py:747` |
| `input_sample_rate` | ✅ Yes | From `TranscriptionMetrics.input_sample_rate` - `backend/transcribe.py:645, processor.py:748` |
| `input_format` | ✅ Yes | From `TranscriptionMetrics.input_format` - `backend/transcribe.py:647, processor.py:750` |
| `requested_language` | ✅ Yes | From processor's language field - `backend/processor.py:729` |
| `applied_language` | ✅ Yes | From `TranscriptionMetrics.applied_language` - `backend/transcribe.py:624, processor.py:730` |
| `detected_language` | ✅ Yes | From `TranscriptionMetrics.detected_language` - `backend/transcribe.py:625, processor.py:731` |
| `language_probability` | ✅ Yes | From `TranscriptionMetrics.language_probability` - `backend/transcribe.py:626, processor.py:732` |

**Implementation details:**
- All statistics come from `TranscriptionMetrics` objects created in transcription functions
- `TranscriptionMetrics` is populated in:
  - `backend/transcribe.py:transcribe()` (main function)
  - `backend/variants/executor.py:transcribe_with_baseline_params()`
  - `backend/variants/executor.py:transcribe_with_minimal_params()`
- Values flow: TranscriptionMetrics → FileMetricRecord → database

## File Level Parameters (file_metrics table)

Both file-level parameters in `FILE_PARAMETER_COLUMNS` are properly set:

| Parameter | Set? | Source |
|-----------|------|--------|
| `output_format` | ✅ Yes | From processor's output_format - `backend/processor.py:788` |
| `float_precision` | ✅ Yes | From `TranscriptionMetrics.float_precision` (FLOAT_PRECISION=3) - `backend/transcribe.py:702, processor.py:789` |

## Optional Statistics (may legitimately be NULL)

Statistics in `OPTIONAL_FILE_STATISTICS` are handled correctly:

| Statistic | Calculated? | Notes |
|-----------|-------------|-------|
| `preprocess_snr_before` | ✅ Yes (may be None) | From `TranscriptionMetrics.preprocess_snr_before` - `backend/transcribe.py:638, processor.py:742`<br>May be None if SNR estimation wasn't performed |
| `preprocess_snr_after` | ✅ Yes (may be None) | From `TranscriptionMetrics.preprocess_snr_after` - `backend/transcribe.py:639, processor.py:743`<br>May be None if SNR estimation wasn't performed |
| `input_bit_depth` | ❌ **NOT calculated** | **Always None** - `backend/transcribe.py:646, processor.py:749`<br>**Reason**: AudioInfo class doesn't extract bit depth from ffprobe<br>**Code comment**: "Not available in current AudioInfo implementation"<br>**Note**: ffprobe command requests `bit_rate` but it's not used; `sample_fmt` is different from bit depth |

## Key Findings

1. **All required statistics are calculated and stored correctly** - The codebase properly populates all expected statistics except for `input_bit_depth`.

2. **`input_bit_depth` is intentionally not available**:
   - The `AudioInfo` class (`backend/preprocess/io.py:13-17`) only extracts:
     - `channels`
     - `sample_rate`
     - `duration`
     - `sample_format` (not the same as bit depth)
   - While ffprobe is requested to show `bit_rate` in the command (`backend/preprocess/io.py:36`), this value is never extracted or used
   - The code explicitly sets `input_bit_depth=None` in all transcription functions
   - This is documented in the code comments

3. **Statistics flow path**:
   ```
   Transcription → TranscriptionMetrics → FileMetricRecord → Database
   ```

4. **Aggregation path** (job level):
   ```
   Individual FileMetrics → Sum/Average → RunRecord → Database
   ```

## Recommendations

1. **`input_bit_depth`**: 
   - If this statistic is needed, the `AudioInfo` class would need to be enhanced to extract bit depth from ffprobe
   - Alternatively, bit depth could be derived from `sample_fmt` (e.g., "s16" = 16-bit, "s32" = 32-bit, "fltp" = 32-bit float)
   - For now, it's correctly excluded from required statistics in `OPTIONAL_FILE_STATISTICS`

2. **No action needed**: All other statistics are properly implemented and stored.



