from .swin_transformer3 import SwinTransformerUnet

class SwinUnet(SwinTransformerUnet):
    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        spatial_resolution,
        **kwargs,
    ):
        in_chans = dim_in 
        num_output_fields = dim_out

        super().__init__(
            in_chans=in_chans,
            num_output_fields=num_output_fields,
            **kwargs,
        )