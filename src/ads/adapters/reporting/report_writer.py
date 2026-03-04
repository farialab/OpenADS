"""Report writing adapter.

Handles file I/O for writing text reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class ReportWriter:
    """Write reports to disk.

    Pure I/O adapter - writes text files.
    """

    def write_text_report(
        self,
        report_text: str,
        output_path: Path,
        encoding: str = "utf-8",
    ) -> Path:
        """Write text report to file.

        Args:
            report_text: Report content as string
            output_path: Path where report will be saved
            encoding: Text encoding (default utf-8)

        Returns:
            Path to written file
        """
        output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(output_path, 'w', encoding=encoding) as f:
            f.write(report_text)

        return output_path

    def append_to_report(
        self,
        additional_text: str,
        output_path: Path,
        encoding: str = "utf-8",
    ) -> Path:
        """Append text to existing report.

        Args:
            additional_text: Text to append
            output_path: Path to existing report
            encoding: Text encoding

        Returns:
            Path to updated file
        """
        output_path = Path(output_path)

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to file
        with open(output_path, 'a', encoding=encoding) as f:
            f.write(additional_text)

        return output_path

    def write_multiple_reports(
        self,
        reports: Dict[str, str],
        output_dir: Path,
        encoding: str = "utf-8",
    ) -> Dict[str, Path]:
        """Write multiple reports to directory.

        Args:
            reports: Dict mapping filename -> report content
            output_dir: Directory for output files
            encoding: Text encoding

        Returns:
            Dict mapping filename -> written file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written_paths = {}
        for filename, content in reports.items():
            output_path = output_dir / filename
            self.write_text_report(content, output_path, encoding)
            written_paths[filename] = output_path

        return written_paths
