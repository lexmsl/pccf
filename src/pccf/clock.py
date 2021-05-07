import time
import sys
from datetime import datetime


while True:
    try:
        now = datetime.now()
        print("%s/%s/%s %s:%s:%s" % (now.month,
                                     now.day,
                                     now.year,
                                     now.hour,
                                     now.minute,
                                     now.second), flush=True)
        time.sleep(1)
    except KeyboardInterrupt:
        print('\nBuy')
        sys.exit(0)
