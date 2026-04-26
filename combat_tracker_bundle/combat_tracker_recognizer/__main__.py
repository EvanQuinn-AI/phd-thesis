"""Module entrypoint: ``python -m combat_tracker_recognizer review <session_id>``."""

import sys

from combat_tracker_recognizer.review.cli import main as review_main


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "review":
        return review_main(sys.argv[2:])
    print("usage: python -m combat_tracker_recognizer review <session_id> [--db PATH] [--parent CLASS]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
