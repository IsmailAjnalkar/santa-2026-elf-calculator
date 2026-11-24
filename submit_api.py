"""Submit a file to santa-2026 over IPv4 with a socket timeout.
usage: python submit_api.py <file> <message>
This network's IPv6 route to www.googleapis.com (Kaggle's upload host) stalls in the TLS handshake,
and the kaggle CLI uploads with no timeout, so a plain `kaggle competitions submit` hangs forever."""
import socket, sys
import urllib3.util.connection as uc

uc.allowed_gai_family = lambda: socket.AF_INET   # IPv4 only
socket.setdefaulttimeout(120)                    # never hang forever

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()
print(api.competition_submit(sys.argv[1], sys.argv[2], 'santa-2026', quiet=True))
print('Successfully submitted')
