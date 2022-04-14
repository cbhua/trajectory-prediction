import torch
import torch_scatter
from torch_scatter import scatter_mean, scatter_max, scatter_add
import inspect

# Aggregate function
def scatter_(name, src, index, device, dim_size=None):
    assert name in ['add', 'mean', 'max']
    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name == 'max' else 0
    out = op(src, index.to(device), 0, None, dim_size)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out

class MessagePassing(torch.nn.Module):
    def __init__(self, device, aggr='add'):
        super(MessagePassing, self).__init__()
        self.device = device
        self.message_args = inspect.getargspec(self.message)[0][1:] # 除去最开始的参数
        # getargspec returns ArgSpec(args, varargs, keywords, defaults)
        self.update_args = inspect.getargspec(self.update)[0][2:] # 除去前两个参数

    def propagate(self, aggr, edge_index, **kwargs):
        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index
        # Collect message arguments
        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])
        # Collect update arguments
        update_args = [kwargs[arg] for arg in self.update_args]
        # Collect message from message arguments
        out = self.message(*message_args)
        # Aggregated messages
        out = scatter_(aggr, out, edge_index[0], self.device, dim_size=size)
        # Update messages
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def update(self, aggr_out):  # pragma: no cover
        return aggr_out