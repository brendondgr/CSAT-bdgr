import torch
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

class GPUInfo:
    def __init__(self):
        self.no_gpu = False
        self.device = None
        self.is_cuda_avail = torch.cuda.is_available()
        self.is_xpu_avail = torch.xpu.is_available()
        self.total_vram = None
        self.current_vram_usage = None
        self.max_memory_allocated = None

    def print_info(self):
        print(f'#############################################')
        print(f'## GPU INFORMATION ##')

        # Print the Version of Torch/CUDA
        print(f'Torch Version: {torch.__version__}')

        # Check if CUDA is available
        print(f'Is CUDA Available: {self.is_cuda_avail}')
        
        # Check if XPU is available
        print(f'Is XPU Available: {self.is_xpu_avail}')

        if self.is_cuda_avail or self.is_xpu_avail:
            # Set Device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Print the Device.
            if self.is_cuda_avail:
                print(f'Current Device: {torch.cuda.get_device_name()}')
                
                print(f'Number of GPUs: {torch.cuda.device_count()}')
                
                self.total_vram = torch.cuda.get_device_properties(0).total_memory
                print(f'Total VRAM: {self.total_vram / (1024 ** 3):.2f} GB')
                
                self.current_vram_usage = torch.cuda.memory_allocated()
                print(f'Current VRAM Usage: {self.current_vram_usage / (1024 ** 3):.2f} GB')
                
                self.max_memory_allocated = torch.cuda.max_memory_allocated()
                print(f'Max Memory Allocated: {self.max_memory_allocated / (1024 ** 3):.2f} GB')
            elif self.is_xpu_avail:
                print(f'Current Device: {torch.xpu.get_device_name()}')
                
                print(f'Number of GPUs: {torch.xpu.device_count()}')
                
                self.total_vram = torch.xpu.get_device_properties(0).total_memory
                print(f'Total VRAM: {self.total_vram / (1024 ** 3):.2f} GB')
                
                self.current_vram_usage = torch.xpu.memory_allocated()
                print(f'Current VRAM Usage: {self.current_vram_usage / (1024 ** 3):.2f} GB')
                
                self.max_memory_allocated = torch.xpu.max_memory_allocated()
                print(f'Max Memory Allocated: {self.max_memory_allocated / (1024 ** 3):.2f} GB')
            else:
                self.no_gpu = True
                print(f'No GPU Available')

        print(f'#############################################\n')

if __name__ == '__main__':
    gpu_info = GPUInfo()
    gpu_info.print_info()