#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use metal::Device;
    use metal::{Buffer, MTLResourceOptions};
    use metal_kernels::{
        call_reshape_and_cache, call_reshape_and_cache_metal4, ConstantValues, Kernels,
        MetalKernelError, PagedAttentionDType, Source, Value,
    };

    fn new_buffer_from_slice<T: Copy>(device: &Device, data: &[T]) -> Buffer {
        let size = std::mem::size_of_val(data) as u64;
        let ptr = data.as_ptr() as *const c_void;
        device.new_buffer_with_data(ptr, size, MTLResourceOptions::StorageModeShared)
    }

    fn read_buffer_as_vec_f32(buffer: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buffer.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
    }

    #[test]
    fn test_kernel_creation() {
        // Test that we can create the kernels singleton
        let kernels = Kernels::default();
        // This should not panic and should return the same instance each time
        let kernels2 = Kernels::default();
        assert!(std::ptr::eq(kernels as *const _ as *const Kernels, kernels2 as *const _ as *const Kernels));
    }

    #[test]
    fn test_device_availability() {
        // Test that we can get a Metal device (this will fail on non-Mac platforms)
        if let Some(device) = Device::system_default() {
            println!("Metal device available: {:?}", device.name());
            assert!(!device.name().is_empty());
        } else {
            println!("Metal device not available (expected on non-Mac platforms)");
        }
    }

    #[test]
    fn test_source_enum() {
        // Test that our Source enum works correctly
        let sources = vec![
            Source::CopyBlocks,
            Source::ReshapeAndCache,
            Source::PagedAttention,
        ];

        for source in sources {
            let kernels = Kernels::new();
            let content = kernels.get_library_source(source);
            assert!(!content.is_empty());
            println!("Source {:?} has {} bytes of shader code", source, content.len());
        }
    }

    #[test]
    fn test_paged_attention_dtype() {
        // Test PagedAttentionDType enum
        let dtypes = vec![
            PagedAttentionDType::F16,
            PagedAttentionDType::BF16,
            PagedAttentionDType::F32,
        ];

        for dtype in dtypes {
            let value = dtype as i32;
            assert!(value >= 0 && value <= 2);
        }
    }

    #[test]
    fn test_constant_values() {
        // Test ConstantValues creation and function values generation
        let values = ConstantValues::new(vec![
            (10, Value::Bool(true)),
            (20, Value::Bool(false)),
        ]);

        let func_values = values.function_constant_values();
        // Just verify it doesn't panic
        drop(func_values);
    }

    #[test]
    fn test_value_trait() {
        // Test Value enum traits
        let val1 = Value::Bool(true);
        let val2 = Value::Bool(false);

        // Test Clone
        let val1_clone = val1;
        assert_eq!(val1, val1_clone);

        // Test Copy
        let val1_copy = val1;
        assert_eq!(val1, val1_copy);

        // Test Hash
        use std::collections::HashSet;
        let set: HashSet<Value> = vec![val1, val2].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_error_messages() {
        // Test that our error types display properly
        let error = MetalKernelError::LoadLibraryError("Test error".to_string());
        assert!(error.to_string().contains("Test error"));

        let error = MetalKernelError::LoadFunctionError("Function test".to_string());
        assert!(error.to_string().contains("Function test"));
    }

    #[cfg(feature = "metal4")]
    #[test]
    fn reshape_and_cache_metal4_matches_metal3() {
        let Some(device) = Device::system_default() else {
            return;
        };

        let num_tokens = 2i32;
        let num_heads = 1i32;
        let head_size = 4i32;
        let block_size = 2i32;
        let x = 2i32;
        let num_blocks = 2usize;
        let key_stride = num_heads * head_size;
        let value_stride = num_heads * head_size;

        let key: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0];
        let value: Vec<f32> = vec![101.0, 102.0, 103.0, 104.0, 111.0, 112.0, 113.0, 114.0];
        let slot_mapping: [i64; 2] = [0, 3];

        let key_cache_len =
            (num_blocks as i32 * num_heads * (head_size / x) * block_size * x) as usize;
        let value_cache_len = (num_blocks as i32 * num_heads * head_size * block_size) as usize;
        let key_cache_init = vec![-1.0f32; key_cache_len];
        let value_cache_init = vec![-1.0f32; value_cache_len];

        let queue = device.new_command_queue();
        let kernels = Kernels::default();

        let key_m3 = new_buffer_from_slice(&device, &key);
        let value_m3 = new_buffer_from_slice(&device, &value);
        let slot_m3 = new_buffer_from_slice(&device, &slot_mapping);
        let key_cache_m3 = new_buffer_from_slice(&device, &key_cache_init);
        let value_cache_m3 = new_buffer_from_slice(&device, &value_cache_init);
        let cb_m3 = queue.new_command_buffer();
        call_reshape_and_cache(
            &device,
            cb_m3,
            kernels,
            PagedAttentionDType::F32,
            &key_m3,
            0,
            &value_m3,
            0,
            &key_cache_m3,
            0,
            &value_cache_m3,
            0,
            &slot_m3,
            0,
            num_tokens,
            num_heads,
            head_size,
            block_size,
            x,
            key_stride,
            value_stride,
        )
        .expect("metal3 reshape_and_cache should succeed");
        cb_m3.commit();
        cb_m3.wait_until_completed();

        let key_m4 = new_buffer_from_slice(&device, &key);
        let value_m4 = new_buffer_from_slice(&device, &value);
        let slot_m4 = new_buffer_from_slice(&device, &slot_mapping);
        let key_cache_m4 = new_buffer_from_slice(&device, &key_cache_init);
        let value_cache_m4 = new_buffer_from_slice(&device, &value_cache_init);
        let cb_m4 = queue.new_command_buffer();
        call_reshape_and_cache_metal4(
            &device,
            cb_m4,
            kernels,
            PagedAttentionDType::F32,
            &key_m4,
            0,
            &value_m4,
            0,
            &key_cache_m4,
            0,
            &value_cache_m4,
            0,
            &slot_m4,
            0,
            num_tokens,
            num_heads,
            head_size,
            block_size,
            x,
            key_stride,
            value_stride,
        )
        .expect("metal4 reshape_and_cache should succeed");
        cb_m4.commit();
        cb_m4.wait_until_completed();

        let got_key_m3 = read_buffer_as_vec_f32(&key_cache_m3, key_cache_len);
        let got_key_m4 = read_buffer_as_vec_f32(&key_cache_m4, key_cache_len);
        let got_val_m3 = read_buffer_as_vec_f32(&value_cache_m3, value_cache_len);
        let got_val_m4 = read_buffer_as_vec_f32(&value_cache_m4, value_cache_len);

        assert_eq!(got_key_m4, got_key_m3, "key_cache mismatch");
        assert_eq!(got_val_m4, got_val_m3, "value_cache mismatch");
    }

    #[cfg(feature = "metal4")]
    #[test]
    fn reshape_and_cache_metal4_perf_smoke() {
        let Some(device) = Device::system_default() else {
            return;
        };

        let num_tokens = 256i32;
        let num_heads = 8i32;
        let head_size = 64i32;
        let block_size = 16i32;
        let x = 8i32;
        let num_blocks = 64usize;
        let key_stride = num_heads * head_size;
        let value_stride = num_heads * head_size;

        let elem_count = (num_tokens * key_stride) as usize;
        let key: Vec<f32> = (0..elem_count).map(|i| i as f32 * 0.001).collect();
        let value: Vec<f32> = (0..elem_count).map(|i| 1000.0 + i as f32 * 0.001).collect();
        let slot_mapping: Vec<i64> = (0..num_tokens).map(|i| i as i64).collect();

        let key_cache_len =
            (num_blocks as i32 * num_heads * (head_size / x) * block_size * x) as usize;
        let value_cache_len = (num_blocks as i32 * num_heads * head_size * block_size) as usize;
        let key_cache_init = vec![0.0f32; key_cache_len];
        let value_cache_init = vec![0.0f32; value_cache_len];

        let queue = device.new_command_queue();
        let kernels = Kernels::default();

        let run = |use_metal4: bool| {
            let key_buf = new_buffer_from_slice(&device, &key);
            let value_buf = new_buffer_from_slice(&device, &value);
            let slot_buf = new_buffer_from_slice(&device, &slot_mapping);
            let key_cache_buf = new_buffer_from_slice(&device, &key_cache_init);
            let value_cache_buf = new_buffer_from_slice(&device, &value_cache_init);

            // Warmup pipeline creation.
            let warmup_cb = queue.new_command_buffer();
            if use_metal4 {
                call_reshape_and_cache_metal4(
                    &device,
                    warmup_cb,
                    kernels,
                    PagedAttentionDType::F32,
                    &key_buf,
                    0,
                    &value_buf,
                    0,
                    &key_cache_buf,
                    0,
                    &value_cache_buf,
                    0,
                    &slot_buf,
                    0,
                    num_tokens,
                    num_heads,
                    head_size,
                    block_size,
                    x,
                    key_stride,
                    value_stride,
                )
                .expect("metal4 warmup should succeed");
            } else {
                call_reshape_and_cache(
                    &device,
                    warmup_cb,
                    kernels,
                    PagedAttentionDType::F32,
                    &key_buf,
                    0,
                    &value_buf,
                    0,
                    &key_cache_buf,
                    0,
                    &value_cache_buf,
                    0,
                    &slot_buf,
                    0,
                    num_tokens,
                    num_heads,
                    head_size,
                    block_size,
                    x,
                    key_stride,
                    value_stride,
                )
                .expect("metal3 warmup should succeed");
            }
            warmup_cb.commit();
            warmup_cb.wait_until_completed();

            let start = std::time::Instant::now();
            for _ in 0..200 {
                let cb = queue.new_command_buffer();
                if use_metal4 {
                    call_reshape_and_cache_metal4(
                        &device,
                        cb,
                        kernels,
                        PagedAttentionDType::F32,
                        &key_buf,
                        0,
                        &value_buf,
                        0,
                        &key_cache_buf,
                        0,
                        &value_cache_buf,
                        0,
                        &slot_buf,
                        0,
                        num_tokens,
                        num_heads,
                        head_size,
                        block_size,
                        x,
                        key_stride,
                        value_stride,
                    )
                    .expect("metal4 run should succeed");
                } else {
                    call_reshape_and_cache(
                        &device,
                        cb,
                        kernels,
                        PagedAttentionDType::F32,
                        &key_buf,
                        0,
                        &value_buf,
                        0,
                        &key_cache_buf,
                        0,
                        &value_cache_buf,
                        0,
                        &slot_buf,
                        0,
                        num_tokens,
                        num_heads,
                        head_size,
                        block_size,
                        x,
                        key_stride,
                        value_stride,
                    )
                    .expect("metal3 run should succeed");
                }
                cb.commit();
                cb.wait_until_completed();
            }
            start.elapsed()
        };

        let t_m3 = run(false);
        let t_m4 = run(true);
        println!("perf-smoke reshape_and_cache: metal3={t_m3:?} metal4={t_m4:?}");

        let ratio = t_m4.as_secs_f64() / t_m3.as_secs_f64();
        assert!(
            ratio <= 1.5,
            "metal4 path is catastrophically slower in smoke test: ratio={ratio:.3}"
        );
    }
}
