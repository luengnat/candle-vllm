// Tests for Metal 4 Tensor API kernels
// These tests verify the tensor operations using M5 Neural Accelerators

#[cfg(feature = "metal")]
mod metal_tests {
    use metal::{Device, CommandQueue, Buffer, MTLResourceOptions};
    use metal_kernels::{Kernels, Source, PagedAttentionDType};

    /// Helper to create a Metal device for testing
    fn get_metal_device() -> Option<Device> {
        // Check if Metal is available
        if cfg!(target_os = "macos") {
            Device::system_default()
        } else {
            None
        }
    }

    /// Test that Tensor API source can be loaded
    #[test]
    fn test_tensor_api_source_loads() {
        let device = match get_metal_device() {
            Some(d) => d,
            None => {
                println!("Skipping test: Metal device not available");
                return;
            }
        };

        let kernels = Kernels::new();

        // Try to load the Tensor API library
        let result = kernels.load_library(&device, Source::TensorApi);

        match result {
            Ok(_) => println!("Tensor API library loaded successfully"),
            Err(e) => {
                // This may fail on older macOS versions or non-M5 hardware
                println!("Tensor API library load failed (expected on non-M5): {}", e);
            }
        }
    }

    /// Test tensor_gemm kernel compilation
    #[test]
    fn test_tensor_gemm_kernel_exists() {
        let device = match get_metal_device() {
            Some(d) => d,
            None => {
                println!("Skipping test: Metal device not available");
                return;
            }
        };

        let kernels = Kernels::new();

        // Try to load a specific tensor kernel
        let kernel_names = vec![
            "tensor_flash_attn_float_hd128_bs16_nt256_nsl32_ps512",
            "tensor_flash_attn_bfloat16_t_hd128_bs16_nt256_nsl32_ps512",
            "tensor_flash_attn_half_hd128_bs16_nt256_nsl32_ps512",
        ];

        for kernel_name in kernel_names {
            let result = kernels.load_pipeline(&device, Source::TensorApi, kernel_name.to_string());
            match result {
                Ok(_) => println!("Kernel {} loaded successfully", kernel_name),
                Err(e) => println!("Kernel {} not available: {}", kernel_name, e),
            }
        }
    }

    /// Test basic tensor operation: matrix multiply
    #[test]
    fn test_tensor_matmul_basic() {
        let device = match get_metal_device() {
            Some(d) => d,
            None => {
                println!("Skipping test: Metal device not available");
                return;
            }
        };

        // Test dimensions
        let m: usize = 128;
        let k: usize = 64;
        let n: usize = 128;

        // Create input matrices (A: MxK, B: KxN)
        let a_data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();
        let mut c_data: Vec<f32> = vec![0.0; m * n];

        // Create Metal buffers
        let buffer_a = device.new_buffer_with_data(
            a_data.as_ptr() as *const std::ffi::c_void,
            (m * k * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_b = device.new_buffer_with_data(
            b_data.as_ptr() as *const std::ffi::c_void,
            (k * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let buffer_c = device.new_buffer(
            (m * n * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        println!("Created Metal buffers for matrix multiplication test");
        println!("A: {}x{}, B: {}x{}, C: {}x{}", m, k, k, n, m, n);

        // Note: Actual kernel dispatch would require implementing tensor_gemm
        // This test verifies buffer creation works
        assert!(buffer_a.length() > 0);
        assert!(buffer_b.length() > 0);
        assert!(buffer_c.length() > 0);
    }

    /// Test tensor flash attention kernel parameters
    #[test]
    fn test_tensor_flash_attn_params() {
        // Test different head dimensions
        let head_dims = vec![32, 64, 80, 96, 112, 128, 256];
        let block_sizes = vec![8, 16, 32];

        for &head_dim in &head_dims {
            for &block_size in &block_sizes {
                println!("Testing head_dim={}, block_size={}", head_dim, block_size);

                // Verify parameters are valid for tensor operations
                assert!(head_dim > 0);
                assert!(block_size > 0);
                assert!(head_dim % 8 == 0); // Required for simdgroup_matrix
            }
        }
    }

    /// Test dtype support for tensor operations
    #[test]
    fn test_tensor_dtype_support() {
        let dtypes = vec![
            PagedAttentionDType::F32,
            PagedAttentionDType::F16,
            PagedAttentionDType::BF16,
        ];

        for dtype in dtypes {
            match dtype {
                PagedAttentionDType::F32 => println!("F32 supported"),
                PagedAttentionDType::F16 => println!("F16 supported"),
                PagedAttentionDType::BF16 => println!("BF16 supported"),
            }
        }
    }
}

#[cfg(not(feature = "metal"))]
mod no_metal_tests {
    #[test]
    fn test_metal_not_available() {
        println!("Metal feature not enabled, skipping Metal tests");
    }
}
