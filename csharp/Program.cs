namespace test
{
    using System.Text;
    using System;
    using Cuda;

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

        public static void Gpu()
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
