from __future__ import annotations

import argparse
from pathlib import Path


def write_c_array(model_path: Path, symbol: str, out_path: Path) -> Path:
    blob = Path(model_path).read_bytes()
    lines = [
        f"extern const unsigned char {symbol}[] __attribute__((aligned(16))) = {{",
    ]
    for start in range(0, len(blob), 16):
        lines.append("  " + ", ".join("0x%02x" % b for b in blob[start:start + 16]) + ",")
    lines.append("};")
    lines.append(f"extern const unsigned int {symbol}_len = {len(blob)};")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = write_c_array(args.model, args.symbol, args.out)
    print(f"wrote {out} for {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
