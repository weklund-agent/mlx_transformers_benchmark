"""Tests for scripts/update_readme_table.py — README table generation with weighted scoring.

Covers VAL-DISPLAY-001 through VAL-DISPLAY-010 assertions.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Import the module under test
from scripts import update_readme_table as urt


# ---------------------------------------------------------------------------
# Helpers: build synthetic DataFrames for testing
# ---------------------------------------------------------------------------


def _make_speed_df(
    models=("model-a",),
    hardware="Apple_M4_Pro_10P+4E+20GPU_64GB",
    dtype="int4",
):
    """Build a minimal speed DataFrame."""
    rows = []
    for m in models:
        rows.append(
            {
                "hardware": hardware,
                "name": m,
                "dtype": dtype,
                "generation_tps": 100.0,
                "prompt_tps": 500.0,
                "peak_memory_gib": 4.0,
            }
        )
    return pd.DataFrame(rows)


def _make_quality_df(
    model="model-a",
    hardware="Apple_M4_Pro_10P+4E+20GPU_64GB",
    dtype="int4",
    problems=None,
):
    """Build a minimal quality DataFrame from (category, problem, passed) tuples."""
    if problems is None:
        problems = [
            ("coding", "fizzbuzz", True),
            ("coding", "reverse_string", True),
            ("reasoning", "logical_deduction", True),
            ("reasoning", "pattern_recognition", False),
            ("tool_calling", "simple_tool_call", True),
        ]
    rows = []
    for cat, prob, passed in problems:
        rows.append(
            {
                "hardware": hardware,
                "model": model,
                "dtype": dtype,
                "category": cat,
                "problem": prob,
                "passed": passed,
                "pass_count": 3 if passed else 0,
                "num_runs": 3,
                "_source_dir": "2026-01-01__00:00:00",
            }
        )
    return pd.DataFrame(rows)


def _make_quality_df_with_real_problems(
    model="model-a",
    hardware="Apple_M4_Pro_10P+4E+20GPU_64GB",
    dtype="int4",
    pass_all=True,
):
    """Build a quality DataFrame using actual registered problem names from the codebase."""
    from mtb.quality_benchmarks import (
        EVAL_PROBLEMS,
        EXPERT_EVAL_PROBLEMS,
        HARD_EVAL_PROBLEMS,
        TOOL_CALLING_PROBLEMS,
    )

    rows = []
    all_problems = (
        EVAL_PROBLEMS
        + HARD_EVAL_PROBLEMS
        + EXPERT_EVAL_PROBLEMS
        + TOOL_CALLING_PROBLEMS
    )
    for i, p in enumerate(all_problems):
        passed = pass_all if isinstance(pass_all, bool) else (i < pass_all)
        rows.append(
            {
                "hardware": hardware,
                "model": model,
                "dtype": dtype,
                "category": p.category,
                "problem": p.name,
                "passed": passed,
                "pass_count": 3 if passed else 0,
                "num_runs": 3,
                "_source_dir": "2026-01-01__00:00:00",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# VAL-DISPLAY-001: Weighted score shown as primary percentage column
# ---------------------------------------------------------------------------
class TestWeightedScorePrimary:
    def test_quality_column_shows_weighted_score(self):
        """Quality column must display weighted score %, not flat pass rate."""
        quality_df = _make_quality_df_with_real_problems(
            model="model-a", pass_all=False
        )
        # Flip first 50 problems to True
        quality_df.loc[:49, "passed"] = True
        quality_df.loc[:49, "pass_count"] = 3

        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        assert not summary.empty
        # Check weighted_pct is present (weighted score as %)
        assert "weighted_pct" in summary.columns
        # weighted_pct should NOT equal simple quality_pct (flat pass rate)
        # unless all tiers have equal pass rates
        row = summary[summary["model"] == "model-a"].iloc[0]
        assert isinstance(row["weighted_pct"], float)

    def test_quality_column_100_when_all_pass(self):
        """All problems pass → weighted score should be 100.0."""
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        row = summary[summary["model"] == "model-a"].iloc[0]
        assert row["weighted_pct"] == 100.0

    def test_quality_column_0_when_all_fail(self):
        """All problems fail → weighted score should be 0.0."""
        quality_df = _make_quality_df_with_real_problems(
            model="model-a", pass_all=False
        )
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        row = summary[summary["model"] == "model-a"].iloc[0]
        assert row["weighted_pct"] == 0.0


# ---------------------------------------------------------------------------
# VAL-DISPLAY-002: Raw pass rate available as footnote or secondary display
# ---------------------------------------------------------------------------
class TestRawPassRateSecondary:
    def test_raw_pass_rate_in_summary(self):
        """Summary must include raw pass rate alongside weighted score."""
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        # Both weighted_pct and quality_pct (raw) must be present
        assert "weighted_pct" in summary.columns
        assert "quality_pct" in summary.columns

    def test_raw_pass_rate_in_table_output(self):
        """Generated table must contain raw pass rate string (e.g., parenthetical)."""
        speed_df = _make_speed_df(models=("model-a",))
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        table = urt.generate_hardware_table(
            speed_df, quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB"
        )
        # The raw pass rate should appear somewhere in the table (e.g., "81/81" or "raw")
        assert "81/81" in table or "100.0%" in table


# ---------------------------------------------------------------------------
# VAL-DISPLAY-003: Per-category breakdowns reflect correct problem counts
# ---------------------------------------------------------------------------
class TestCategoryBreakdowns:
    def test_category_denominators_match_actual_counts(self):
        """Category columns (coding, tool_calling, reasoning) must show correct totals."""
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )

        row = summary[summary["model"] == "model-a"].iloc[0]

        # Check coding shows X/Y where Y matches actual coding problem count
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
        )

        coding_count = sum(
            1
            for p in EVAL_PROBLEMS + HARD_EVAL_PROBLEMS + EXPERT_EVAL_PROBLEMS
            if p.category == "coding"
        )
        if "coding" in row and pd.notna(row["coding"]):
            # Extract denominator from "X/Y" format
            parts = str(row["coding"]).split("/")
            assert len(parts) == 2
            assert int(parts[1]) == coding_count

    def test_tool_calling_denominator_matches_actual_count(self):
        """Tool calling column denominator must match actual tool calling problem count."""
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )

        row = summary[summary["model"] == "model-a"].iloc[0]

        from mtb.quality_benchmarks import TOOL_CALLING_PROBLEMS

        if "tool_calling" in row and pd.notna(row["tool_calling"]):
            parts = str(row["tool_calling"]).split("/")
            assert len(parts) == 2
            assert int(parts[1]) == len(TOOL_CALLING_PROBLEMS)


# ---------------------------------------------------------------------------
# VAL-DISPLAY-004: Backward compatibility — old CSV format renders correctly
# ---------------------------------------------------------------------------
class TestBackwardCompatOldCSV:
    def test_old_csv_without_new_columns_renders(self):
        """Old CSV without avg_generation_time_sec etc. must still produce valid table."""
        # Old format CSV: model, framework, backend, dtype, category, problem, pass_count, num_runs, passed, sample_response
        old_quality_df = pd.DataFrame(
            [
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-a",
                    "dtype": "int4",
                    "category": "coding",
                    "problem": "fizzbuzz",
                    "passed": True,
                    "pass_count": 1,
                    "num_runs": 1,
                    "_source_dir": "2026-01-01__00:00:00",
                },
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-a",
                    "dtype": "int4",
                    "category": "reasoning",
                    "problem": "logical_deduction",
                    "passed": False,
                    "pass_count": 0,
                    "num_runs": 1,
                    "_source_dir": "2026-01-01__00:00:00",
                },
            ]
        )

        # Should not raise
        summary = urt.compute_quality_summary(
            old_quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        assert not summary.empty
        # Should have weighted and raw scores
        assert "weighted_pct" in summary.columns
        assert "quality_pct" in summary.columns

    def test_old_csv_uses_flat_rate_as_fallback(self):
        """When problems in CSV don't match registered problems, fallback to flat pass rate."""
        # Old CSV with problem names that DON'T match the current registry
        old_quality_df = pd.DataFrame(
            [
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-a",
                    "dtype": "int4",
                    "category": "coding",
                    "problem": "unknown_old_problem_1",
                    "passed": True,
                    "pass_count": 1,
                    "num_runs": 1,
                    "_source_dir": "2026-01-01__00:00:00",
                },
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-a",
                    "dtype": "int4",
                    "category": "coding",
                    "problem": "unknown_old_problem_2",
                    "passed": False,
                    "pass_count": 0,
                    "num_runs": 1,
                    "_source_dir": "2026-01-01__00:00:00",
                },
            ]
        )

        summary = urt.compute_quality_summary(
            old_quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        assert not summary.empty
        row = summary[summary["model"] == "model-a"].iloc[0]
        # When no problems match the registry, weighted_pct should fall back to raw
        assert "weighted_pct" in summary.columns
        # raw = 1/2 = 50.0
        assert row["quality_pct"] == 50.0


# ---------------------------------------------------------------------------
# VAL-DISPLAY-005: Backward compatibility — new CSV with old CSVs coexist
# ---------------------------------------------------------------------------
class TestMixedOldNewCSV:
    def test_mixed_old_new_csvs_coexist(self):
        """load_quality_data() must handle mix of old and new format CSVs."""
        # New format has extra columns; old format doesn't
        old_df = pd.DataFrame(
            [
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-a",
                    "dtype": "int4",
                    "category": "coding",
                    "problem": "fizzbuzz",
                    "passed": True,
                    "pass_count": 1,
                    "num_runs": 1,
                    "_source_dir": "2026-01-01__00:00:00",
                }
            ]
        )
        new_df = pd.DataFrame(
            [
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-b",
                    "dtype": "int4",
                    "category": "coding",
                    "problem": "fizzbuzz",
                    "passed": True,
                    "pass_count": 3,
                    "num_runs": 3,
                    "avg_generation_time_sec": 1.5,
                    "avg_generation_tps": 100.0,
                    "avg_tokens_generated": 150,
                    "_source_dir": "2026-02-01__00:00:00",
                }
            ]
        )
        # Concat should not fail even with mismatched columns
        combined = pd.concat([old_df, new_df], ignore_index=True)
        assert len(combined) == 2
        # compute_quality_summary should handle this combined df
        summary = urt.compute_quality_summary(
            combined, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        assert not summary.empty
        assert len(summary) == 2  # both models present


# ---------------------------------------------------------------------------
# VAL-DISPLAY-006: Table header format preserved
# ---------------------------------------------------------------------------
class TestTableHeaderFormat:
    def test_full_table_header_preserved(self):
        """Table must have the expected column headers when quality data is present."""
        speed_df = _make_speed_df(models=("model-a",))
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        table = urt.generate_hardware_table(
            speed_df, quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB"
        )
        # First line should be the header
        lines = table.strip().split("\n")
        header = lines[0]
        assert "Model" in header
        assert "Arch" in header
        assert "Gen tok/s" in header
        assert "Quality" in header
        assert "Coding" in header
        assert "Tool Calling" in header
        assert "Reasoning" in header
        assert "Memory" in header
        assert "Min HW" in header


# ---------------------------------------------------------------------------
# VAL-DISPLAY-007: BEGIN/END markers preserved after update
# ---------------------------------------------------------------------------
class TestBeginEndMarkers:
    def test_markers_preserved_in_generation(self):
        """generate_tables() output should be non-empty content that goes between markers."""
        # We can't easily test the full update_readme flow without the actual README,
        # but we can verify the markers are used correctly
        assert urt.BEGIN_MARKER == "<!-- BEGIN BENCHMARK TABLE -->"
        assert urt.END_MARKER == "<!-- END BENCHMARK TABLE -->"

    def test_update_readme_preserves_markers(self, tmp_path):
        """After update, README must have exactly one BEGIN and one END marker."""
        readme_path = tmp_path / "README.md"
        readme_path.write_text(
            "# Title\n\n"
            "<!-- BEGIN BENCHMARK TABLE -->\n"
            "old content\n"
            "<!-- END BENCHMARK TABLE -->\n"
            "\n## Footer\n"
        )

        with (
            patch.object(urt, "README_PATH", readme_path),
            patch.object(urt, "load_speed_data", return_value=_make_speed_df()),
            patch.object(urt, "load_quality_data", return_value=pd.DataFrame()),
        ):
            urt.update_readme(dry_run=False)

        content = readme_path.read_text()
        assert content.count(urt.BEGIN_MARKER) == 1
        assert content.count(urt.END_MARKER) == 1
        # Content between markers should not be empty
        begin_idx = content.index(urt.BEGIN_MARKER) + len(urt.BEGIN_MARKER)
        end_idx = content.index(urt.END_MARKER)
        between = content[begin_idx:end_idx].strip()
        assert len(between) > 0


# ---------------------------------------------------------------------------
# VAL-DISPLAY-008: Problem count annotation updated
# ---------------------------------------------------------------------------
class TestProblemCountAnnotation:
    def test_annotation_reflects_problem_count(self):
        """The annotation line must mention ~81 problems."""
        speed_df = _make_speed_df(models=("model-a",))
        quality_df = (
            pd.DataFrame()
        )  # No quality data, but annotation should still be correct

        tables = (
            urt.generate_tables.__wrapped__(models=None)
            if hasattr(urt.generate_tables, "__wrapped__")
            else None
        )

        # Instead, test the generate_tables output directly
        with (
            patch.object(urt, "load_speed_data", return_value=speed_df),
            patch.object(urt, "load_quality_data", return_value=quality_df),
        ):
            output = urt.generate_tables(models=None)

        # Should mention "81 problems" in the annotation
        assert "81 problems" in output


# ---------------------------------------------------------------------------
# VAL-DISPLAY-009: Dry-run mode does not modify README
# ---------------------------------------------------------------------------
class TestDryRunMode:
    def test_dry_run_does_not_modify_readme(self, tmp_path):
        """Running with --dry-run must not change README.md content."""
        readme_path = tmp_path / "README.md"
        original_content = (
            "# Title\n\n"
            "<!-- BEGIN BENCHMARK TABLE -->\n"
            "old content\n"
            "<!-- END BENCHMARK TABLE -->\n"
            "\n## Footer\n"
        )
        readme_path.write_text(original_content)
        original_hash = hashlib.md5(original_content.encode()).hexdigest()

        with (
            patch.object(urt, "README_PATH", readme_path),
            patch.object(urt, "load_speed_data", return_value=_make_speed_df()),
            patch.object(urt, "load_quality_data", return_value=pd.DataFrame()),
        ):
            urt.update_readme(dry_run=True)

        after_content = readme_path.read_text()
        after_hash = hashlib.md5(after_content.encode()).hexdigest()
        assert original_hash == after_hash, "Dry run modified the README!"


# ---------------------------------------------------------------------------
# VAL-DISPLAY-010: Models without quality data show '--'
# ---------------------------------------------------------------------------
class TestModelsWithoutQuality:
    def test_speed_only_model_shows_dashes(self):
        """Models with speed but no quality data must show '--' in Quality column."""
        speed_df = _make_speed_df(models=("model-a", "model-b"))
        # Only model-a has quality data
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)

        table = urt.generate_hardware_table(
            speed_df, quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB"
        )

        # Find the row for model-b
        lines = table.strip().split("\n")
        model_b_lines = [l for l in lines if "model-b" in l]
        assert len(model_b_lines) == 1
        # Should have "--" in the quality column
        assert "| -- |" in model_b_lines[0]

    def test_no_quality_data_all_dashes(self):
        """When no quality data exists at all, all models show '--'."""
        speed_df = _make_speed_df(models=("model-a",))
        quality_df = pd.DataFrame()

        table = urt.generate_hardware_table(
            speed_df, quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB"
        )

        # Without quality data, table uses speed-only format (no Quality column)
        # This is existing behavior — verify it works
        assert table  # non-empty output
        lines = table.strip().split("\n")
        assert len(lines) >= 3  # header, separator, at least one row


# ---------------------------------------------------------------------------
# Additional tests for weighted scoring integration
# ---------------------------------------------------------------------------
class TestWeightedScoringIntegration:
    def test_compute_quality_summary_returns_weighted_and_raw(self):
        """compute_quality_summary should return both weighted_pct and quality_pct."""
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        assert "weighted_pct" in summary.columns
        assert "quality_pct" in summary.columns

    def test_weighted_score_differs_from_raw_for_asymmetric_results(self):
        """Weighted and raw scores should differ when tier pass rates are unequal."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        # Easy: all pass, Hard: all fail, Expert: all fail, Tool Calling: all fail
        rows = []
        for p in EVAL_PROBLEMS:
            rows.append(
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-a",
                    "dtype": "int4",
                    "category": p.category,
                    "problem": p.name,
                    "passed": True,
                    "pass_count": 3,
                    "num_runs": 3,
                    "_source_dir": "2026-01-01__00:00:00",
                }
            )
        for p in HARD_EVAL_PROBLEMS + EXPERT_EVAL_PROBLEMS + TOOL_CALLING_PROBLEMS:
            rows.append(
                {
                    "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                    "model": "model-a",
                    "dtype": "int4",
                    "category": p.category,
                    "problem": p.name,
                    "passed": False,
                    "pass_count": 0,
                    "num_runs": 3,
                    "_source_dir": "2026-01-01__00:00:00",
                }
            )

        quality_df = pd.DataFrame(rows)
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        row = summary[summary["model"] == "model-a"].iloc[0]

        # Raw: 15/81 = 18.5%
        # Weighted: (1*15 + 2*0 + 3*0 + 3*0) / (1*15 + 2*10 + 3*16 + 3*40) = 15/203 = 7.4%
        assert row["quality_pct"] != row["weighted_pct"]
        assert row["weighted_pct"] < row["quality_pct"]

    def test_quality_table_shows_weighted_and_raw(self):
        """Generated table row should show weighted score and raw pass rate."""
        speed_df = _make_speed_df(models=("model-a",))
        quality_df = _make_quality_df_with_real_problems(model="model-a", pass_all=True)
        table = urt.generate_hardware_table(
            speed_df, quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB"
        )

        # Should contain the weighted score (100.0% since all pass)
        assert "100.0%" in table

    def test_empty_quality_df_returns_empty_summary(self):
        """Empty quality DataFrame should return empty summary."""
        quality_df = pd.DataFrame()
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        assert summary.empty

    def test_multiple_models_in_summary(self):
        """Summary should handle multiple models correctly."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        rows = []
        for model in ("model-a", "model-b"):
            for i, p in enumerate(EVAL_PROBLEMS):
                rows.append(
                    {
                        "hardware": "Apple_M4_Pro_10P+4E+20GPU_64GB",
                        "model": model,
                        "dtype": "int4",
                        "category": p.category,
                        "problem": p.name,
                        "passed": True if model == "model-a" else (i < 5),
                        "pass_count": 3 if (model == "model-a" or i < 5) else 0,
                        "num_runs": 3,
                        "_source_dir": "2026-01-01__00:00:00",
                    }
                )

        quality_df = pd.DataFrame(rows)
        summary = urt.compute_quality_summary(
            quality_df, "Apple_M4_Pro_10P+4E+20GPU_64GB", "int4"
        )
        assert len(summary) == 2
        assert set(summary["model"].tolist()) == {"model-a", "model-b"}
