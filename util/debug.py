IS_DEBUG = True

def debug(*args, **kwargs):
    if IS_DEBUG:
        print(*args, **kwargs, flush=True)