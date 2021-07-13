import networkx as nx
import metis
import matplotlib.pyplot as plt
import onnx
from onnx import helper
from onnx import shape_inference, AttributeProto, TensorProto, GraphProto
import numpy as np

# DSL

## variable
class Variable:
    def __init__(
            self, graph, name,
            type, shape):
        self.graph = graph
        self.name = name
        self.type = type
        self.shape = shape
        self.slices = [(0, dim) for dim in shape]
        self.producers = []

    def slice(self, *ranges):
        assert(len(self.shape) == len(ranges))
        v = Variable(
                self.graph, self.name,
                self.type, self.shape)
        slices = []
        for ((o_f, o_t), (n_f, n_t)) in zip(self.slices, ranges):
            assert(o_t-o_f >= n_t-n_f)
            slices.append((o_f+n_f, o_f+n_t))
        v.slices = slices
        return v


def variable(graph, name, type=None, shape=[]):
    if name not in graph.graph['variables']:
        assert(type != None)
        assert(shape != [])
        v = Variable(graph, name, type, shape)
        graph.graph['variables'][name] = v
    return graph.graph['variables'][name]


## helper
def is_overlay(slices_a, slices_b):
    assert(len(slices_a) == len(slices_b))
    for ((a_f, a_t), (b_f, b_t)) in zip(slices_a, slices_b):
        if b_f > a_t-1 or a_f > b_t-1:
            return False
    return True

## vertex
class Vertex:
    def __init__(
            self, 
            graph, id,
            inputs, outputs, cost, func):
        self.graph = graph
        self.id = id
        self.cost = cost
        self.inputs = inputs
        self.outputs = outputs
        self.connections = {}


    def __setitem__(self, port, var):
        assert((port in self.inputs) or (port in self.outputs))
        self.connections[port] = (var.name, var.slices)
        n = self.graph.nodes[self.id]
        if port in self.inputs:
            for v in variable(self.graph, var.name).producers:
                if is_overlay(v[1], var.slices):
                    self.graph.add_edge(v[0].id, self.id)
            if 'inputs' not in n:
                n['inputs'] = []
            n['inputs'].append(var)
        else:
            variable(self.graph, var.name).producers.append((self, var.slices))
            if 'outputs' not in n:
                n['outputs'] = []
            n['outputs'].append(var)


def vertex(graph, inputs, outputs, cost, func):
    id = len(graph.nodes)
    graph.add_node(
        id,
        func=func,
        weight=cost,
    )
    v = Vertex(
            graph, id,
            inputs, outputs, cost, func)
    return v


# operator implementations
def matmul(graph, node):
    lhs = variable(graph, node.input[0])
    rhs = variable(graph, node.input[1])
    output = variable(graph, node.output[0])
    def compute(data):
        lhs = data[node.input[0]]
        rhs = data[node.input[1]]
        result = np.matmul(lhs, rhs)
        print("run matmul tile")
        return [result]
    # split lhs on dim0 and split rhs on dim1
    for i in range(0, lhs.shape[0]):
        for j in range(0, rhs.shape[1]):
            v = vertex(graph, node.input, node.output, 5, compute)
            v[node.input[0]] = lhs.slice((i, i+1), (0, lhs.shape[1]))
            v[node.input[1]] = rhs.slice((0, rhs.shape[0]), (j, j+1))
            v[node.output[0]] = output.slice((i, i+1), (j, j+1))


def add(graph, node):
    shape = variable(graph, node.output[0]).shape
    def compute(data):
        lhs = data[node.input[0]]
        rhs = data[node.input[1]]
        result = np.add(lhs, rhs)
        print("run add tile")
        return [result]
    # split on dim0 and dim1
    for i in range(0, shape[0]):
        step = shape[1]
        for j in range(0, int(shape[1]/step)):
            v = vertex(graph, node.input, node.output, 1, compute)
            for input in node.input:
                v[input] = variable(graph, input).slice((i, i+1), (j, j+step))
            for output in node.output:
                v[output] = variable(graph, output).slice((i, i+1), (j, j+step))


# convertor
def onnx2graph(model):
    g = nx.DiGraph()
    g.graph['variables'] = {}
    model = shape_inference.infer_shapes(model)
    def add_var(node):
        return variable(
                g, node.name,
                node.type.tensor_type.elem_type,
                [dim.dim_value for dim in \
                        node.type.tensor_type.shape.dim])
    # inputs
    for node in model.graph.input:
        add_var(node)
    # internal tensors
    for node in model.graph.value_info:
        add_var(node)
    # outputs
    for node in model.graph.output:
        add_var(node)
    # computing nodes
    for node in model.graph.node:
        if node.op_type == 'MatMul':
            matmul(g, node)
        elif node.op_type == 'Add':
            add(g, node)
    return g


# target
class Target:
    def __init__(self):
        self.cores = ['red','blue','green','yellow']
        self.sram_size = 256 * 1024


# merge nodes so that each cluster has similar cost
def merge_nodes(graph, target):
    graph.graph['node_weight_attr'] = 'weight'
    # here clusters may has cycles after partition
    # need a customized alg
    cuts, parts = metis.part_graph(graph.to_undirected(), len(target.cores))
    for i, p in enumerate(parts):
        graph.nodes[i]['color'] = target.cores[p]
    return graph, parts


# check if clusters is a dag
def get_clusters_dag(graph, parts, target):
    dag = nx.DiGraph()
    mapping = {}
    for i, p in enumerate(parts):
        if p not in dag.nodes:
            dag.add_node(target.cores[p])
        mapping[i] = target.cores[p]
    for f, t in graph.edges:
        if mapping[f] != mapping[t]:
            dag.add_edge(mapping[f], mapping[t])
    assert(nx.is_directed_acyclic_graph(dag))
    return dag


# set step number for each node
def set_step_number(graph, cluster_dag):
    dag = cluster_dag.copy()
    step = 0
    while len(dag) > 0:
        colors = [color for color, d in dag.in_degree() if d == 0]
        for color in colors:
            for node in graph.nodes():
                if graph.nodes[node]['color'] == color:
                    graph.nodes[node]['step'] = step
            dag.remove_node(color)
        step += 1
    graph.graph['total_steps'] = step


def bytes_of_type(type):
    if type == TensorProto.FLOAT:
        return 4
    assert(0)

# allocate memory and
# check if total memory usage exceeds sram size
# acutally we can reuse memory in dag, but for now
# forget about it.
def alloc_memory(graph, parts, target):
    sizes = {}
    types = {}
    for i, p in enumerate(parts):
        if p not in sizes:
            sizes[p] = {}
            types[p] = {}
        n = graph.nodes[i]
        for output in n['outputs']:
            if output.name not in sizes[p]:
                sizes[p][output.name] = output.slices
                types[p][output.name] = output.type
            sizes[p][output.name] = \
                    [(min(a_f, b_f), max(a_t, b_t)) \
                    for ((a_f, a_t), (b_f, b_t)) in \
                    zip(sizes[p][output.name], output.slices)]
        for input in n['inputs']:
            if input.name not in sizes[p]:
                sizes[p][input.name] = input.slices
                types[p][input.name] = input.type
            sizes[p][input.name] = \
                    [(min(a_f, b_f), max(a_t, b_t)) \
                    for ((a_f, a_t), (b_f, b_t)) in \
                    zip(sizes[p][input.name], input.slices)]
    graph.graph['buffers'] = {}
    for _, p in enumerate(sizes):
        size = 0
        graph.graph['buffers'][target.cores[p]] = sizes[p]
        for _, a in enumerate(sizes[p]):
            s = 1
            for (f, t) in sizes[p][a]:
                s *= (t-f)
            size += s*bytes_of_type(types[p][a])
        assert(size <= target.sram_size)


# placement and routing
# clusters can be placed to help on latency of data movement
# do nothing here so the virtual core is just phyiscal
def place_and_route(graph):
    return graph


# compile
def compile_onnx(model, target):
    g = onnx2graph(model)
    g, clusters = merge_nodes(g, target)
    cluster_dag = get_clusters_dag(g, clusters, target)
    set_step_number(g, cluster_dag)
    alloc_memory(g, clusters, target)
    g = place_and_route(g)
    return g


# show
def show(graph):
    colors = []
    for node in graph:
        if 'color' in graph.nodes[node]:
            colors.append(graph.nodes[node]['color'])
        else:
            colors.append('yellow')
    nx.draw(graph, node_color=colors, with_labels=True)
    plt.show()


# simulator
def compute_steps(graph):
    return graph.graph['total_steps']

def slices_in_buffer(buffer_slices, slices):
    return [(s_f-b_f, s_t-b_f) for ((b_f, b_t), (s_f, s_t)) \
            in zip(buffer_slices, slices)]

def to_np_slices(slices):
    return [range(f, t) for (f, t) in slices]

def vertex_compute(graph, target, step, data):
    _, clusters = merge_nodes(graph, target)
    cluster_dag = get_clusters_dag(graph, clusters, target)
    order = nx.topological_sort(cluster_dag)
    buffers = graph.graph['buffers']
    for cluster in order:
        for node in nx.topological_sort(graph):
            n = graph.nodes[node]
            if n['step'] <= step and n['color'] == cluster:
                print("run node {} in step {}".format(node, n['step']))
                assert(cluster in data)
                inputs = {}
                for input in n['inputs']:
                    inputs[input.name] = \
                            data[cluster][input.name][ \
                            to_np_slices(
                                slices_in_buffer(
                                    buffers[cluster][input.name],
                                    input.slices))]
                    print(inputs[input.name])
                results = n['func'](inputs)
                print(results)
                for (output, result) in zip(n['outputs'], results):
                    data[cluster][output.name][\
                            to_np_slices(
                                slices_in_buffer(
                                    buffers[cluster][output.name],
                                    output.slices))] = result

def data_transfer(graph, target, step, data):
    for node in nx.topological_sort(graph):
        n = graph.nodes[node]
        print("data transfer in step {} for node {}".format(n['step'], node))
        cluster = n['color']
        buffers = graph.graph['buffers']
        if cluster not in data:
            data[cluster] = {}
        for input in n['inputs']:
            assert(input.name in data)
            if input.name not in data[cluster]:
                data[cluster][input.name] = np.zeros(
                        [t-f for (f, t) in buffers[cluster][input.name]], np.float32)
            data[cluster][input.name][\
                    to_np_slices(
                        slices_in_buffer(
                            buffers[cluster][input.name],
                            input.slices))] = \
                    data[input.name][to_np_slices(input.slices)]
        for output in n['outputs']:
            if output.name not in data[cluster]:
                data[cluster][output.name] = np.zeros(
                        [t-f for (f, t) in buffers[cluster][output.name]], np.float32)
            if output.name not in data:
                data[output.name] = np.zeros(output.shape, np.float32)
            data[output.name][to_np_slices(output.slices)] = \
                    data[cluster][output.name][\
                    to_np_slices(
                        slices_in_buffer(
                            buffers[cluster][output.name],
                            output.slices))]

def sim(graph, target, inputs):
    data = inputs
    for step in range(0, compute_steps(graph)):
        data_transfer(graph, target, step, data)
        vertex_compute(graph, target, step, data)
    data_transfer(graph, target, compute_steps(graph), data)


# test

# create onnx model
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [4, 2])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [2, 8])
Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [4, 8])

node_matmul = helper.make_node(
    'MatMul',
    ['X', 'Y'],
    ['T'],
)

node_add = helper.make_node(
    'Add',
    ['T', 'T'],
    ['Z'],
)

graph_def = helper.make_graph(
    [node_matmul, node_add],
    'test-model',
    [X, Y],
    [Z],
)

model_def = helper.make_model(graph_def, producer_name='matmul-add')
onnx.checker.check_model(model_def)

target = Target()
graph = compile_onnx(model_def, target)

show(graph)

x_data = np.random.rand(4, 2)
y_data = np.random.rand(2, 8)
data = {'X': x_data, 'Y': y_data}
sim(graph, target, data)
result=data['Z']
print(data['Z'])
expected = np.matmul(x_data, y_data)
expected = np.add(expected, expected)
print(expected)
np.testing.assert_allclose(result, expected, rtol=1e-5, atol=0)

