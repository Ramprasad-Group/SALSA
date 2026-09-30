"""Print the first TCP port, starting at a given one, that nothing listens on.

A port counts as taken if any listener holds it on localhost (IPv4 or IPv6)
or on all interfaces. Checking only one address is not enough: an SSH
LocalForward on 127.0.0.1:8501 coexists with a server on *:8501 and steals
every http://localhost:8501 request.

Usage: python find_free_port.py [start_port] [max_tries]
"""
from __future__ import annotations

import errno
import socket
import sys

_ADDRESSES = [
    (socket.AF_INET, "127.0.0.1"),
    (socket.AF_INET, "0.0.0.0"),
    (socket.AF_INET6, "::1"),
    (socket.AF_INET6, "::"),
]


def port_is_free(port: int) -> bool:
    for family, host in _ADDRESSES:
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            continue  # e.g. IPv6 disabled on this machine
        try:
            # No SO_REUSEADDR, so binding fails if any listener overlaps.
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((host, port))
        except OSError as exc:
            # EADDRNOTAVAIL means the address itself is unusable here (e.g.
            # no ::1), not that the port is taken.
            if exc.errno == errno.EADDRNOTAVAIL:
                continue
            return False
        finally:
            sock.close()
    return True


def main() -> int:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    tries = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    for port in range(start, start + tries):
        if port_is_free(port):
            print(port)
            return 0
    print(f"No free port in {start}-{start + tries - 1}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
