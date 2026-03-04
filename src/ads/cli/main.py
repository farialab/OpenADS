"""Unified CLI router for OpenADS."""

from __future__ import annotations

import sys
from typing import Sequence

from . import combined, dwi, pwi


def _print_help() -> None:
    print(
        "OpenADS CLI\n"
        "\n"
        "Usage:\n"
        "  ads <command> [args]\n"
        "  ads help <command>\n"
        "\n"
        "Commands:\n"
        "  dwi        Run DWI pipeline\n"
        "  pwi        Run PWI pipeline\n"
        "  combined   Run DWI + PWI combined workflow\n"
        "  help       Show top-level help or command help\n"
        "  version    Show CLI version\n"
        "\n"
        "Get Detailed Help:\n"
        "  ads help dwi\n"
        "  ads help pwi\n"
        "  ads help combined\n"
        "\n"
        "Quick Examples:\n"
        "  ads dwi --subject-path assets/examples/dwi/sub-02e8eb42 --all\n"
        "  ads pwi --subject-path assets/examples/pwi/sub-02e8eb42 --stages prepdata,inference,report\n"
        "  ads combined --dwi-subject-path assets/examples/dwi/sub-02e8eb42 --pwi-subject-path assets/examples/pwi/sub-02e8eb42 --all\n"
        "\n"
        "Notes:\n"
        "  - Run from project root if using relative paths.\n"
        "  - Use --config to override default pipeline YAML files.\n"
        "  - Command-specific options are documented by 'ads help <command>'.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    commands = {
        "dwi": dwi.run,
        "pwi": pwi.run,
        "combined": combined.run,
    }

    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    cmd = args[0].lower()
    rest = args[1:]

    if cmd == "help":
        if rest and rest[0].lower() in commands:
            return commands[rest[0].lower()](["--help"])
        _print_help()
        return 0

    if cmd in {"version", "--version", "-V"}:
        print("OpenADS CLI 1.0.0")
        return 0

    if cmd in commands:
        return commands[cmd](rest)

    print(f"Unknown command: {args[0]}")
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
