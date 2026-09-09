IS_DEBUG = False

def debug(*args, **kwargs):
    if IS_DEBUG:
        print(*args, **kwargs, flush=True)