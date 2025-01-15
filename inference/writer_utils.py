import os, re, torch
from typing import Literal
from datetime import datetime

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
        self.write_flag = False
        self.path = "log/tmp.log"
        self.title = "Empty Title"
        self.dir_path = os.path.dirname(self.path)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def __del__(self):
        self.stop()

    def write(self, line):
        if self.write_flag:
            self.file.write(f"{line}\n")

    def stop(self):
        self.write_flag = False
        try:
            self.file.flush()
            self.file.close()
        except:
            pass

    def clear(self):
        self.stop()
        self.file = open(self.path, "w+")
        self.write(self.title)
        self.write_flag = True


class XCCL_Writer(Writer):
    def __init__(self, rank = -1):
        super().__init__(rank)
        if rank < 0:
            return
        self.path = f"log/model_hccl_{rank}.csv"
        self.title = "Layer,Algo,DataShape,DataType,DataSize"
        self.clear()

    def recoder(self, layer, algo, tensor):
        if self.write_flag:
            self.write(f"{layer},{algo},{get_tensor_info(tensor)}")


class FLOPs_Writer(Writer):
    def __init__(self, rank = -1):
        super().__init__(rank)
        if rank < 0:
            return
        self.t0 = datetime.now()
        self.total_cube_flops = 0
        self.total_vector_flops = 0
        self.path = f"log/model_flops_{rank}.csv"
        self.title = "Layer,Ops,MatA_Shape,MatA_Type,MatA_Size,MatB_Shape,MatB_Type,MatB_Size,Bias_Shape,Bias_Type,Bias_Size,Scale_Shape,Scale_Type,Scale_Size,Cube_Dim,Cube_FLOPs,Vector_FLOPs"
        self.clear()

    def print(self, flush_only):
        speed = flops = dur = cube_flops = vector_flops = 0
        if not flush_only and self.write_flag and self.rank == 0:
            dur = (datetime.now() - self.t0).total_seconds()
            flops = self.total_cube_flops + self.total_vector_flops
            speed = get_unit(flops / dur)
            flops = get_unit(flops)
            cube_flops = get_unit(self.total_cube_flops)
            vector_flops = get_unit(self.total_vector_flops)
            print(f"Model: {speed}FLOPs/xpu ({flops}op/xpu in {dur:.2f}s), Cube: {cube_flops}op/xpu, Vector: {vector_flops}op/xpu")
        self.t0 = datetime.now()
        self.total_cube_flops = 0
        self.total_vector_flops = 0
        return speed, flops, dur, cube_flops, vector_flops

    def recoder(self, layer, ops, mat_a, mat_b = None, bias = None, scale = None):
        if self.write_flag:
            cube_flops = 0
            vector_flops = 0
            cube_dim = 1
            if re.match(r".*Embed.*", ops, re.IGNORECASE):
                pass
            elif re.match(r".*(einsum|GEMM).*", ops, re.IGNORECASE):
                mat_a_flops = float(torch.prod(torch.tensor(mat_a.shape)))
                mat_b_flops = float(torch.prod(torch.tensor(mat_b.shape)))
                cube_dim = str(mat_a.shape).split(",")[-1].replace(")","").replace("]","")
                if cube_dim not in str(mat_b.shape):
                    raise Exception(f"Cube dim {cube_dim} not in mat_b shape {mat_b.shape}")
                cube_flops = mat_a_flops * mat_b_flops / float(cube_dim)
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
        self.t0 = datetime.now()
        self.path = f"log/model_memory_{rank}.csv"
        self.title = "Time,Dur_s,Layer,Stage,Ops,Ops_Type,Memory_in_Used,DataShape,DataType,DataSize"
        self.clear()

    def recoder(self, layer, stage: Literal["init", "forward", "cache"], ops: Literal["malloc", "free"], op_type, tensor):
        if self.write_flag and tensor is not None:
            if stage in ["init", "forward"]:
                if ops == "malloc":
                    self.memory_used += get_tensor_size(tensor)
                elif ops == "free":
                    self.memory_used -= get_tensor_size(tensor)
                else:
                    raise Exception(f"Unsupported ops: {ops}")
            self.write(f"{datetime.now()},{(datetime.now()- self.t0).total_seconds()},{layer},{ops},{op_type},{self.memory_used},{get_tensor_info(tensor)}")


class Weights_Writer(Writer):
    def __init__(self, rank = -1):
        super().__init__(rank)
        if rank < 0:
            return
        self.path = f"log/model_model_{rank}.csv"
        self.title = "DataName,DataShape,DataType,DataSize"
        self.clear()

    def recoder(self, model):
        if self.write_flag:
            model_info = model.state_dict()
            for item in model_info:
                self.write(f"{item},{get_tensor_info(model_info[item])}")
            print(model)
