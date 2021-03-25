import numpy as np


class Tensor(object):
    """Node in a computation graph."""

    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.
            
            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object, 
                e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Tensor):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        if isinstance(other, Tensor):
            new_node = mul_op(self, other)
        else:
            new_node = mul_byconst_op(self, other)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            new_node = div_op(self, other)
        else:
            new_node = div_byconst_op(self, other)
        return new_node

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            new_node = div_op(self, other)
        else:
            new_node = rdiv_byconst_op(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Tensor):
            new_node = sub_op(self, other)
        else:
            new_node = sub_byconst_op(self, other)
        return new_node

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            new_node = sub_op(self, other)
        else:
            new_node = rsub_byconst_op(self, other)
        return new_node

    def __neg__(self):
        return neg_op(self)

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


def Variable(name) -> object:
    """User defined variables in an expression.  
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    return placeholder_node


def Constant(value, name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    constant_node = constant_op(value)
    constant_node.name = name
    return constant_node


class Op(object):
    """Op represents operations performed on nodes."""

    def __call__(self, *args, **kwargs):
        """Create a new node and associate the op object with the node.
        
        Returns
        -------
        The new node object.
        """
        new_node = Tensor()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        assert False, "Implemented in subclass"

    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        assert False, "Implemented in subclass"


class NegOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "-%s" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return -input_vals[0]

    def gradient(self, node, output_grad):
        return [-output_grad]


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contributions to each input."""
        return [output_grad, output_grad]


class SubOp(Op):

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "%s-%s" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad, -output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of add node, return gradient contribution to input."""
        return [output_grad]


class SubByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] - node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]


class RSubByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const_attr - input_vals[0]

    def gradient(self, node, output_grad):
        return [-output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input."""
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]


class DivOp(Op):

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "%s/%s" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] / input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[1], -output_grad * node.inputs[0] / (node.inputs[1] * node.inputs[1])]


class DivByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "%s/%s" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] / node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad / node.const_attr]


class RDivByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.const_attr = const_val
        new_node.name = "%s/%s" % (str(const_val), node_A.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return node.const_attr / input_vals[0]

    def gradient(self, node, output_grad):
        return [-output_grad * node.const_attr / (node.inputs[0] * node.inputs[0])]


class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        """TODO: Your code here"""
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        """Given gradient of multiplication node, return gradient contribution to input."""
        """TODO: Your code here"""
        return [output_grad * node.const_attr]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        mat_A = input_vals[0]
        mat_B = input_vals[1]
        if node.matmul_attr_trans_A:
            mat_A = mat_A.T
        if node.matmul_attr_trans_B:
            mat_B = mat_B.T
        return np.matmul(mat_A, mat_B)

    def gradient(self, node, output_grad):
        """Given gradient of multiply node, return gradient contributions to each input.
            
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        return [matmul_op(output_grad, node.inputs[1], False, True),
                matmul_op(node.inputs[0], output_grad, True, False)]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""

    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None


class ConstantOp(Op):

    def __call__(self, value):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        new_node.const_attr = value
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None


class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert (isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        # new_node.name = "array(1)"
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert (isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class LogOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "log(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.log(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad / node.inputs[0]]


class ExpOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "exp(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * exp_op(node.inputs[0])]


class PowOp(Op):

    def __call__(self, node, n):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.n = n
        new_node.name = "pow(%s, %d)" % (node.name, n)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.power(input_vals[0], node.n)

    def gradient(self, node, output_grad):
        return [output_grad * node.n * pow_op(node.inputs[0], node.n - 1)]


class SquareOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "square(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.square(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * 2 * node.inputs[0]]


class ReduceSumOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "reduce_sum(%s)" % node.name
        return new_node

    def compute(self, node, input_vals):
        assert isinstance(input_vals[0], np.ndarray)
        return np.sum(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * oneslike_op(node.inputs[0])]


class CosOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "cos(%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.cos(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * -sin_op(node.inputs[0])]


class SinOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "sin(%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return np.sin(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * cos_op(node.inputs[0])]


class TransposeOp(Op):
    def __call__(self, node, axes=None):
        if axes is None:
            axes = [1, 2, 0, 3]
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.axes = axes
        new_node.name = f"Transpose({node.name})"
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return self.transpose(input_vals[0], node.axes)

    def gradient(self, node, output_grad):
        return [output_grad * self.transpose(node.inputs[0], node.axes)]

    @staticmethod
    def transpose(value, axes):
        return np.transpose(value, axes)


class Rot90Op(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = f"Rot90({node.name})"
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return self.rot90(input_vals[0])

    def gradient(self, node, output_grad):
        return [output_grad * self.rot90(node.inputs[0])]

    @staticmethod
    def rot90(value):
        return value[::-1, ::-1, ...]


class Conv2dOp(Op):
    def __call__(self, node, filter, strides, padding):
        new_node = Op.__call__(self)
        new_node.strides = strides
        new_node.padding = padding
        new_node.inputs = [node, filter]
        new_node.name = f"conv2d({node.name}, {filter.name})"
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return self.conv2d(input_vals[0], input_vals[1], node.strides, node.padding)

    def gradient(self, node, output_grad):
        return [conv2d_op(output_grad, rot90_op(node.inputs[1]), node.strides, "full"),
                conv2d_op(node.inputs[0], transpose_op(output_grad, axes=[1, 2, 0, 3]), node.strides, "valid")]

    @staticmethod
    def conv2d(val, filter, strides, padding):
        def numpy_conv(inputs, filter, strides=None, _result=None):
            if strides is None:
                strides = [1, 1]
            filter_h, filter_w = filter.shape
            strides_h, strides_w = strides
            # 这里先定义一个和输入一样的大空间，但是周围一圈后面会截掉
            result = np.zeros_like(_result)
            # 更新下新输入,SAME模式下，会改变HW
            H, W = inputs.shape
            # print("new size",H,W)
            # 卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
            for r in range(result.shape[0]):
                for c in range(result.shape[1]):
                    # 池化大小的输入区域
                    cur_input = inputs[r * strides_h:r * strides_h + filter_h, c * strides_w:c * strides_w + filter_w]
                    # 和核进行乘法计算
                    cur_output = cur_input * filter
                    # 再把所有值求和
                    conv_sum = np.sum(cur_output)
                    # 当前点输出值
                    result[r, c] = conv_sum
            return result

        def _conv(inputs, filter, strides, padding="valid"):
            strides = strides[1:3]
            batch, H, W, C_in = inputs.shape
            filter_h, filter_w, C_in, C_out = filter.shape
            # C_out指核对个数，也是最后结果对通道个数
            C_out = filter.shape[3]
            # 同样我们任务核对宽高相等
            if padding == "valid":
                result = np.zeros([batch, int(np.floor((H - filter_h + 2 * 0) / strides[0] + 1)),
                                   int(np.floor((W - filter_w + 2 * 0) / strides[1] + 1)), C_out],
                                  np.float32)
            elif padding == "same":
                result = np.zeros([batch, H, W, C_out], np.float32)
                b, H_new, W_new, C = inputs.shape
                pad_h = (H_new - 1) * strides[0] + filter_h - H
                pad_top = int(pad_h / 2)
                pad_down = pad_h - pad_top

                pad_w = (W_new - 1) * strides[1] + filter_w - W
                pad_left = int(pad_w / 2)
                pad_right = pad_w - pad_left
                inputs = np.pad(inputs, ((0, 0), (pad_top, pad_down), (pad_left, pad_right), (0, 0)), 'constant',
                                constant_values=(0, 0))
            else:
                result = np.zeros([batch,
                                   int(np.ceil((H - filter_h + 2 * (filter_h - 1)*strides[0]) / strides[0] + 1)),
                                   int(np.ceil((W - filter_w + 2 * (filter_w - 1)*strides[1]) / strides[1] + 1)),
                                   C_out], np.float32)
                pad_h = (result.shape[1] - 1) * strides[0] - H + filter_h
                pad_top = 1
                pad_down = pad_h - pad_top

                pad_w = (result.shape[2] - 1) * strides[1] - W + filter_w
                pad_left = 1
                pad_right = pad_w - pad_left
                inputs = np.pad(inputs, ((0, 0), (pad_top, pad_down), (pad_left, pad_right), (0, 0)), 'constant',
                                constant_values=(0, 0))
            # 核个数对循环
            for b in range(batch):
                for channel_out in range(C_out):
                    # 输入通道数对循环
                    for channel_in in range(C_in):
                        # 当前通道对数据
                        channel_data = inputs[b, ..., channel_in]
                        # 采用上面对逻辑，单核单通道卷积,然后累计
                        result[b, :, :, channel_out] += numpy_conv(channel_data, filter[..., channel_in, channel_out],
                                                                   strides=strides,
                                                                   _result=result[b, :, :, channel_out])
            return result

        return _conv(val, filter, strides, padding)


# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
div_op = DivOp()
sub_op = SubOp()
neg_op = NegOp()
add_byconst_op = AddByConstOp()
rsub_byconst_op = RSubByConstOp()
sub_byconst_op = SubByConstOp()
mul_byconst_op = MulByConstOp()
div_byconst_op = DivByConstOp()
rdiv_byconst_op = RDivByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
constant_op = ConstantOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
log_op = LogOp()
exp_op = ExpOp()
square_op = SquareOp()
pow_op = PowOp()
sin_op = SinOp()
cos_op = CosOp()
reduce_sum = ReduceSumOp()
conv2d_op = Conv2dOp()
rot90_op = Rot90Op()
transpose_op = TransposeOp()


def matmul(val, n):
    if isinstance(val, Tensor):
        return matmul_op(val, n)
    return np.matmul(val, n)


def pow(val, n):
    if isinstance(val, Tensor):
        return pow_op(val, n)
    return np.power(val, n)


def square(val):
    if isinstance(val, Tensor):
        return square_op(val)
    return np.square(val)


def exp(val):
    if isinstance(val, Tensor):
        return exp_op(val)
    return np.exp(val)


def log(val):
    if isinstance(val, Tensor):
        return log_op(val)
    return np.log(val)


def sin(val):
    if isinstance(val, Tensor):
        return sin_op(val)
    return np.sin(val)


def cos(val):
    if isinstance(val, Tensor):
        return cos_op(val)
    return np.cos(val)


def conv2d(val, filter, strides=None, padding="valid"):
    if strides is None:
        strides = [1, 1, 1, 1]
    if isinstance(val, Tensor):
        return conv2d_op(val, filter, strides=strides, padding=padding)
    return Conv2dOp.conv2d(val, filter, strides, padding)


class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""

    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list. 
        """
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.

        topo_order = find_topo_sort(self.eval_node_list)
        for node in topo_order:
            if isinstance(node.op, PlaceholderOp):
                continue
            vals = [node_to_val_map[n] for n in node.inputs]
            compute_val = node.op.compute(node, vals)
            node_to_val_map[node] = compute_val if isinstance(compute_val, np.ndarray) else np.array(compute_val)

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list = {}
    # Special note on initializing gradient of output_node as oneslike_op(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    # 从节点到该节点的梯度的映射
    node_to_output_grad = {}
    # 给定我们正在使用梯度wrt的output_node，按反拓扑顺序遍历图。
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad
        for i in range(len(node.inputs)):
            ch = node.inputs[i]
            grads = node.op.gradient(node, grad)
            grads_list = node_to_output_grads_list.get(ch, [])
            grads_list.append(grads[i])
            node_to_output_grads_list[ch] = grads_list

    # Collect results for gradients requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


##############################

####### Helper Methods #######

##############################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


if __name__ == '__main__':
    import tensorflow as tf

    x = Variable(name='x')
    w = Variable(name='w')

    strides = [1, 2, 2, 1]
    y = conv2d(x, w, strides=strides)

    w_grad, x_grad = gradients(y, [w, x])
    executor = Executor([y, w_grad, x_grad])

    # x_val = np.reshape(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float), (1, 3, 3, 1))
    # x_val = np.reshape(np.array([[1,  2,  3,  4,  5],
    #                              [6,  7,  8,  9,  10],
    #                              [11, 12, 13, 14, 15],
    #                              [16, 17, 18, 19, 20],
    #                              [21, 22, 23, 24, 25]
    #                              ],
    #                             dtype=np.float), (1, 6, 6, 1))
    w_val = np.reshape(np.array([[1, 2],
                                 [3, 4]],
                                dtype=np.float), (2, 2, 1, 1))
    np.random.seed(1)
    x_val = np.random.random_integers(0, 3, (1, 5, 5, 1)).astype(np.float64)
    w_val = np.random.random_integers(0, 3, (2, 2, 1, 1)).astype(np.float64)

    a, b, c = executor.run(feed_dict={w: w_val, x: x_val})

    xx = tf.convert_to_tensor(x_val)
    ww = tf.convert_to_tensor(w_val)
    yy = tf.nn.conv2d(xx, ww, strides, padding="VALID")
    ww_grad, xx_grad = tf.gradients(yy, [ww, xx])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        yyy, www_grad, xxx_grad = sess.run([yy, ww_grad, xx_grad])
    aaa, bbb, ccc = np.squeeze(a), np.squeeze(b), np.squeeze(c)
    yyy, www_grad, xxx_grad = np.squeeze(yyy), np.squeeze(www_grad), np.squeeze(xxx_grad)
    print()
