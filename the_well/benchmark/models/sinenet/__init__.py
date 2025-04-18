from .sinenet import sinenet

class SineNet(sinenet):
    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_resolution,
        **kwargs,
    ):
        super().__init__( **kwargs )
