using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace test
{
    using System.Text;
    using System;
    using Cuda;
    using System.Linq;

    class Test
    {
        class Helper
        {
            public static string OutProps(Cuda.CUdevprop props, string name)
            {
                StringBuilder sb = new StringBuilder();
                sb.AppendLine("name: " + name);
                sb.AppendLine("clock rate: " + props.Value.clockRate);
                sb.AppendLine("max threads per block: " + props.Value.maxThreadsPerBlock);
                return sb.ToString();
            }
        }

        public static unsafe void Gpu()
        {
            Cuda.CUcontext ctx = new CUcontext(IntPtr.Zero);
            try
            {
                Cuda.Functions.cuInit(0);

                var res = Cuda.Functions.cuDeviceGetCount(out int count);
                if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();

                for (int deviceID = 0; deviceID < count; ++deviceID)
                {
                    res = Cuda.Functions.cuDeviceGet(out CUdevice device, deviceID);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();

                    byte[] name = new byte[100];
                    res = Cuda.Functions.cuDeviceGetName(name, 100, device);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
                    string n = enc.GetString(name).Replace("\0", "");

                    res = Cuda.Functions.cuDeviceGetProperties(out CUdevprop props, device);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();

                    System.Console.WriteLine("--------");
                    System.Console.WriteLine(Helper.OutProps(props, n));
                }

                {
                    res = Cuda.Functions.cuDeviceGet(out CUdevice device, 0);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    res = Cuda.Functions.cuCtxCreate_v2(out CUcontext cuContext, 0, device);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    string path = Assembly.GetAssembly(typeof(Program)).Location;
                    path = Path.GetDirectoryName(path);
                    path = Path.GetFullPath(path + @"\..\..\..\..");
                    path = path + @"\cuda\x64\Debug\vector-sum.ptx";
                    StreamReader sr = new StreamReader(path);
                    String ptx = sr.ReadToEnd();
                    IntPtr ptr = Marshal.StringToHGlobalAnsi(ptx);
                    res = Cuda.Functions.cuModuleLoadData(out CUmodule module, ptr);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    res = Cuda.Functions.cuModuleGetFunction(out CUfunction helloWorld, module, "VectorSumParallel");
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    int n = 3;
                    int[] a = Enumerable.Range(1, 3).Select(v => v * 3).ToArray();
                    int[] b = Enumerable.Range(1, 3).Select(v => v * 2).ToArray();
                    int[] c = new int[3];
                    res = Cuda.Functions.cuMemAlloc_v2(out CUdeviceptr d_a, n * sizeof(int));
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    res = Cuda.Functions.cuMemAlloc_v2(out CUdeviceptr d_b, n * sizeof(int));
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    res = Cuda.Functions.cuMemAlloc_v2(out CUdeviceptr d_c, n * sizeof(int));
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    var ha = GCHandle.Alloc(a, GCHandleType.Pinned);
                    var hb = GCHandle.Alloc(b, GCHandleType.Pinned);
                    var hc = GCHandle.Alloc(c, GCHandleType.Pinned);
                    var pa = ha.AddrOfPinnedObject();
                    var pb = hb.AddrOfPinnedObject();
                    var pc = hc.AddrOfPinnedObject();
                    res = Cuda.Functions.cuMemcpyHtoD_v2(d_a, pa, sizeof(int) * n);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    res = Cuda.Functions.cuMemcpyHtoD_v2(d_b, pb, sizeof(int) * n);
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    IntPtr[] x = new IntPtr[] { (IntPtr)d_a.Value, (IntPtr)d_b.Value, (IntPtr)d_c.Value, (IntPtr)n };
                    GCHandle handle2 = GCHandle.Alloc(x, GCHandleType.Pinned);
                    IntPtr pointer2 = IntPtr.Zero;
                    pointer2 = handle2.AddrOfPinnedObject();
                    IntPtr[] kp = new IntPtr[] { pointer2 };
                    fixed (IntPtr* kernelParams = kp)
                    {
                        res = Cuda.Functions.cuLaunchKernel(
                            helloWorld,
                            1, 1, 1, // grid has one block.
                            (uint)n, 1, 1, // block has 11 threads.
                            0, // no shared memory
                            default(CUstream),
                            (IntPtr)kernelParams,
                            (IntPtr)IntPtr.Zero
                        );
                    }
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    res = Cuda.Functions.cuMemcpyDtoH_v2(hc.AddrOfPinnedObject(), d_c, n * sizeof(int));
                    if (res.Value != cudaError_enum.CUDA_SUCCESS) throw new Exception();
                    Cuda.Functions.cuCtxDestroy_v2(cuContext);
                    System.Console.WriteLine(String.Join(" ",c));
                }
            }
            finally
            {
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
		    Test.Gpu();
        }
    }
}
