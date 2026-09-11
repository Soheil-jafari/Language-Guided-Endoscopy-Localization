"""The former implementation was not an official EndoMamba integration."""

class OriginalEndoMamba:
    def __init__(self,*args,**kwargs):
        raise NotImplementedError('Official EndoMamba architecture/checkpoint integration is unavailable. Do not label a different backbone as EndoMamba.')

EndoMamba=OriginalEndoMamba
