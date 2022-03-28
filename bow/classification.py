from torch import nn

from . import bow_utils as utils


class PredictionHead(nn.Module):
    def __init__(
            self,
            num_channels,
            num_classes,
            batch_norm=False,
            pred_type="linear",
            pool_type="global_avg",
            pool_params=None,
    ):
        """ Builds a prediction head for the classification task."""

        super(PredictionHead, self).__init__()

        if pred_type != "linear":
            raise NotImplementedError(
                f"Not recognized / supported prediction head type '{pred_type}'."
                f" Currently, only pred_type 'linear' is implemented.")
        self.pred_type = pred_type
        total_num_channels = num_channels

        self.layers = nn.Sequential()
        if pool_type == "none":
            if isinstance(pool_params, int):
                output_size = pool_params
                total_num_channels *= (output_size * output_size)
        elif pool_type == "global_avg":
            self.layers.add_module(
                "pooling", utils.GlobalPooling(type="avg"))
        elif pool_type == "avg":
            assert isinstance(pool_params, (list, tuple))
            assert len(pool_params) == 4
            kernel_size, stride, padding, output_size = pool_params
            total_num_channels *= (output_size * output_size)
            self.layers.add_module(
                "pooling", nn.AvgPool2d(kernel_size, stride, padding))
        elif pool_type == "adaptive_avg":
            assert isinstance(pool_params, int)
            output_size = pool_params
            total_num_channels *= (output_size * output_size)
            self.layers.add_module(
                "pooling", nn.AdaptiveAvgPool2d(output_size))
        else:
            raise NotImplementedError(
                f"Not supported pool_type '{pool_type}'. Valid pooling types: "
                "('none', 'global_avg', 'avg', 'adaptive_avg').")

        assert isinstance(batch_norm, bool)
        if batch_norm:
            # Affine is set to False. So, this batch norm layer does not have
            # any learnable (scale and bias) parameters. It's only purpose is
            # to normalize the features. So, the prediction layer is still
            # linear. It is only used for the Places205 linear classification
            # setting to make it the same as the benchmark code:
            # https://github.com/facebookresearch/fair_self_supervision_benchmark
            self.layers.add_module(
                "batch_norm", nn.BatchNorm2d(num_channels, affine=False))
        self.layers.add_module("flattening", nn.Flatten())

        prediction_layer = nn.Linear(total_num_channels, num_classes)
        prediction_layer.weight.data.normal_(0.0, 0.01)
        prediction_layer.bias.data.fill_(0.0)

        self.layers.add_module("prediction_layer", prediction_layer)

    def forward(self, features):
        return self.layers(features)
