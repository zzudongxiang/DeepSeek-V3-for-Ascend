import os, re, torch
from typing import Literal
from datetime import datetime
import matplotlib.pyplot as plt

dir = "model_info"

def get_unit(value):
    units_index = 0
    units = ["", "K", "M", "G", "T", "P"]
    while value > 1000 and units_index < len(units) - 1:
        value /= 1000.0
        units_index += 1
    return f"{value:.2f} {units[units_index]}"

def get_dtype_size(dtype):
    dtype = str(dtype)
    if dtype in ["complex64", "float64", "int64"]:
        return 8
    elif dtype in ["float", "float32", "int32"]:
        return 4
    elif dtype in ["float16", "bfloat16"]:
        return 2
    elif dtype in ["int8", "float8_e4m3fn"]:
        return 1
    else:
        raise Exception(f"Unsupported data type: {dtype}")

def get_tensor_shape(tensor_shape):
    return re.sub(r"torch\.Size|\(|\)| ", "", str(tensor_shape)).replace(",", "x").strip()

def get_tensor_dtype(tensor_dtype):
    return re.sub(r"torch\.|\<|\>|\"|\'|class| ", "", str(tensor_dtype)).strip()

def get_tensor_size(tensor):
    return float(torch.prod(torch.tensor(tensor.shape)) * get_dtype_size(get_tensor_dtype(tensor.dtype)))

def get_tensor_info(tensor):
    if isinstance(tensor, torch.Tensor):
        return f"{get_tensor_shape(tensor.shape)},{get_tensor_dtype(tensor.dtype)},{get_tensor_size(tensor)}"
    elif tensor is not None:
        dtype = get_tensor_dtype(type(tensor))
        return f'[1],{dtype},{get_dtype_size(dtype)}'
    return ",,"

class Writer:
    def __init__(self, rank):
        self.rank = rank
        self.context = []
    def __del__(self):
        if self.rank >= 0:
            self.finished()

    def write(self, line):
        self.context.append(line)

    def finished(self, mode = "w+"):
        if self.rank >= 0:
            self.dir_path = os.path.dirname(self.path)
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path)
            with open(self.path, mode) as file:
                file.write(f"{self.title}\n" if mode == "w+" else "\n")
                file.write("\n".join(self.context))
        self.context = []


class XCCL_Writer(Writer):
    def __init__(self, rank = -1):
        super().__init__(rank)
        if rank < 0:
            return
        self.path = f"log/{dir}/model_hccl_{rank}.csv"
        self.title = "Layer,Algo,DataShape,DataType,DataSize"

    def recoder(self, layer, algo, tensor):
        if self.rank >= 0:
            self.write(f"{layer},{algo},{get_tensor_info(tensor)}")


class FLOPs_Writer(Writer):
    def __init__(self, rank = -1):
        super().__init__(rank)
        if rank < 0:
            return
        self.t0 = datetime.now()
        self.total_cube_flops = 0
        self.total_vector_flops = 0
        self.path = f"log/{dir}/model_flops_{rank}.csv"
        self.title = "Layer,Ops,MatA_Shape,MatA_Type,MatA_Size,MatB_Shape,MatB_Type,MatB_Size,Bias_Shape,Bias_Type,Bias_Size,Scale_Shape,Scale_Type,Scale_Size,Cube_Dim,Cube_FLOPs,Vector_FLOPs"

    def print(self, flush_only):
        speed = flops = dur = cube_flops = vector_flops = 0
        if not flush_only and self.rank == 0:
            dur = (datetime.now() - self.t0).total_seconds()
            flops = self.total_cube_flops + self.total_vector_flops
            speed = get_unit(flops / dur)
            flops = get_unit(flops)
            cube_flops = get_unit(self.total_cube_flops)
            vector_flops = get_unit(self.total_vector_flops)
            print(f"Model: {speed}FLOPs/xpu ({flops}ops/xpu in {dur:.2f}s), Cube: {cube_flops}ops/xpu, Vector: {vector_flops}ops/xpu")
        self.t0 = datetime.now()
        self.total_cube_flops = 0
        self.total_vector_flops = 0
        return speed, flops, dur, cube_flops, vector_flops

    def recoder(self, layer, ops, mat_a, mat_b = None, bias = None, scale = None):
        if self.rank < 0:
            return
        cube_dim = 1
        cube_flops = vector_flops = 0
        if re.match(r".*Embed.*", ops, re.IGNORECASE):
            pass
        elif re.match(r".*(einsum|GEMM).*", ops, re.IGNORECASE):
            mat_a_flops = float(torch.prod(torch.tensor(mat_a.shape)))
            mat_b_flops = float(torch.prod(torch.tensor(mat_b.shape)))
            cube_dim = str(mat_a.shape).split(",")[-1].replace(")","").replace("]","")
            if cube_dim not in str(mat_b.shape):
                raise Exception(f"Cube dim {cube_dim} not in mat_b shape {mat_b.shape}")
            cube_flops = 2 * mat_a_flops * mat_b_flops / float(cube_dim)
        else:
            vector_flops = float(torch.prod(torch.tensor(mat_a.shape)))
            if mat_b is not None and isinstance(mat_b, torch.Tensor):
                vector_flops = max(vector_flops, float(torch.prod(torch.tensor(mat_b.shape))))
        self.total_cube_flops += cube_flops
        self.total_vector_flops += vector_flops
        self.write(f"{layer},{ops},{get_tensor_info(mat_a)},{get_tensor_info(mat_b)},{get_tensor_info(bias)},{get_tensor_info(scale)},{cube_dim},{cube_flops},{vector_flops}")

class Memory_Writer(Writer):
    def __init__(self, rank = -1):
        super().__init__(rank)
        if rank < 0:
            return
        self.memory_used = 0
        self.kv_cache = 0
        self.t0 = datetime.now()
        self.path = f"log/{dir}/model_memory_{rank}.csv"
        self.title = "Time,Dur_s,Layer,Stage,Ops,Ops_Type,Memory_in_Used,DataShape,DataType,DataSize,KV-Cache"
        self.dir_path = os.path.dirname(self.path)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        with open(self.path, "w+") as file:
            file.write(self.title)

    def recoder(self, layer, stage: Literal["init", "forward", "cache"], ops: Literal["malloc", "free"], op_type, tensor):
        if self.rank >= 0 and tensor is not None:
            if stage in ["init", "forward"]:
                if ops == "malloc":
                    self.memory_used += get_tensor_size(tensor)
                elif ops == "free":
                    self.memory_used -= get_tensor_size(tensor)
                else:
                    raise Exception(f"Unsupported ops: {ops}")
            elif stage in ["cache"]:
                if ops == "malloc":
                    self.kv_cache += get_tensor_size(tensor)
                elif ops == "free":
                    self.kv_cache -= get_tensor_size(tensor)
                else:
                    raise Exception(f"Unsupported ops: {ops}")
            else:
                raise Exception(f"Unsupported stage: {stage}")
            self.write(f"{datetime.now()},{(datetime.now()- self.t0).total_seconds()},{layer},{stage},{ops},{op_type},{self.memory_used},{get_tensor_info(tensor)},{self.kv_cache}")


class Weights_Writer(Writer):
    def __init__(self, rank = -1):
        super().__init__(rank)
        if rank < 0:
            return
        self.path = f"log/{dir}/model_weights_{rank}.csv"
        self.title = "DataName,DataShape,DataType,DataSize"

    def recoder(self, model):
        if self.rank < 0:
            return
        model_info = model.state_dict()
        for item in model_info:
            self.write(f"{item},{get_tensor_info(model_info[item])}")

tensor_hist_dict = {}
def tensor_hist(layer, tenser_name, tensor):
    global tensor_hist_dict
    mat = tensor.flatten().float().numpy(force = True)
    dict_key = f"{layer}-{tenser_name}"
    if dict_key not in tensor_hist_dict:
        tensor_hist_dict[dict_key] = 0
    tensor_hist_dict[dict_key] += 1
    dict_key = f"{dict_key}-{tensor_hist_dict[dict_key]}"
    plt.hist(mat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"{dict_key} >> {get_tensor_shape(tensor.shape)},{get_tensor_dtype(tensor.dtype)}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    fig_path = f"log/tensor_hist/{dict_key}.png"
    fig_dir = os.path.dirname(fig_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_path)

if __name__ == '__main__':
    tensor_hist("writer", "randn", torch.randn((1000, 1000), dtype=torch.bfloat16))
