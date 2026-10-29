import argparse
import datetime
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    sys.exit("pyserial is required: pip install pyserial")

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "logs" / "bench_final.log"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--wait", type=float, default=60.0,
                    help="seconds to wait for the port to appear")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    deadline = time.time() + args.wait
    while not Path(args.port).exists():
        if time.time() > deadline:
            sys.exit("port %s did not appear within %.0f s" % (args.port, args.wait))
        time.sleep(0.2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    with serial.Serial(args.port, args.baud, timeout=0.5) as ser:
        print("# capturing %s for %.0f s -> %s" % (args.port, args.seconds, out))
        end = time.time() + args.seconds
        while time.time() < end:
            line = ser.readline()
            if not line:
                continue
            text = line.decode("utf-8", "replace").rstrip()
            print(text, flush=True)
            lines.append(text)

    out.write_text("# %s\n" % stamp + "\n".join(lines) + "\n")
    print("# wrote %d lines to %s" % (len(lines), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
