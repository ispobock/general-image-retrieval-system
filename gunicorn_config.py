import os
import gevent.monkey
gevent.monkey.patch_all()

loglevel = 'info'
bind = "0.0.0.0:8090"
pidfile = "log/gunicorn.pid"
accesslog = "log/access.log"
errorlog = "log/debug.log"
daemon = False

workers = 16
worker_class = 'gevent'
x_forwarded_for_header = 'X-FORWARDED-FOR'
