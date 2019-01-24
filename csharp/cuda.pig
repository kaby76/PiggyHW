using 'ClangSupport.pig';
using 'Enums.pig';
using 'Structs.pig';
using 'Funcs.pig';
using 'Namespace.pig';
using 'Typedefs.pig';

template CudaClangSupport : ClangSupport
{
    init {{
        namespace_name = "Cuda";
        limit = ".*\\.*GPU.*\\.*";
        var list = new List<string>() {
            "cudaError_enum",
            "CUdevice_attribute_enum",
            "CUjit_option_enum",
            "CUmemAttach_flags_enum",
            "CUjitInputType_enum",
             "CUdevprop",
            "^CUresult$",
            "^CUcontext$",
            "^CUfunction$",
            "^CUlinkState$",
            "^CUmodule$",
            "^CUstream$",
            "^CUdevice$",
            "^CUjit_option$",
            "^CUdeviceptr$",
            "^CUdevprop$",
            "^cuCtxCreate_v2$",
            "^cuCtxDestroy_v2",
            "^cuCtxSynchronize$",
            "^cuDeviceGet$",
            "^cuDeviceGetCount$",
            "^cuDeviceGetName$",
            "^cuDeviceGetPCIBusId$",
            "^cuDeviceGetProperties$",
            "^cuDevicePrimaryCtxReset$",
            "^cuDeviceTotalMem_v2$",
            "^cuGetErrorString$",
            "^cuInit$",
            "^cuLaunchKernel$",
            "^cuLinkComplete$",
            "^cuMemAlloc_v2$",
            "^cuMemcpyDtoH_v2$",
            "^cuMemcpyHtoD_v2$",
            "^cuMemFreeHost$",
            "^cuMemGetInfo_v2$",
            "^cuModuleGetFunction$",
            "^cuModuleGetGlobal_v2$",
            "^cuModuleLoadData$",
           };
        generate_for_only = String.Join("|", list);
        dllname = "nvcuda";
    }}
}

template CudaFuncs : Funcs
{
    init {{
        details = new List<generate_type>()
            {
                { new generate_type()
                    {
                        name = ".*",
                        convention = System.Runtime.InteropServices.CallingConvention.Cdecl,
                        special_args = null
                    }
                }
            }; // default for everything.
    }}

    pass Functions {
        ( FunctionDecl SrcRange=$"{ClangSupport.limit}" Name="cuModuleLoadDataEx"
            [[ [DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "cuModuleLoadDataEx")]
            public static extern CUresult cuModuleLoadDataEx(out CUmodule jarg1, IntPtr jarg2, uint jarg3, CUjit_option[] jarg4, IntPtr jarg5);
            
            ]]
        )
        ( FunctionDecl SrcRange=$"{ClangSupport.limit}" Name="cuLaunchKernel"
            [[ [DllImport("nvcuda", CallingConvention = CallingConvention.Cdecl, EntryPoint = "cuLaunchKernel")]
            public static extern CUresult cuLaunchKernel(CUfunction f, uint gridDimX, uint gridDimY, uint gridDimZ, uint blockDimX, uint blockDimY, uint blockDimZ, uint sharedMemBytes, CUstream hStream, IntPtr kernelParams, IntPtr extra);
            
            ]]
        )
    }
}

application
    CudaClangSupport.Start
	Namespace.GenerateStart
    Enums.GenerateEnums
    Typedefs.GeneratePointerTypes
    Structs.GenerateStructs
    Typedefs.GenerateTypedefs
    CudaFuncs.Start
    CudaFuncs.Functions
    CudaFuncs.End
    Namespace.GenerateEnd
    ;
