#[cfg(feature = "metal")]
use candle_core::{MetalDevice, Result};
#[cfg(all(feature = "metal4", target_os = "macos", target_arch = "aarch64"))]
use std::sync::OnceLock;

#[cfg(feature = "metal")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalBackend {
    Metal3,
    Metal4,
}

#[cfg(feature = "metal")]
fn metal4_is_supported(device: &MetalDevice) -> bool {
    #[cfg(feature = "metal4")]
    {
        let _ = device;
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }
    #[cfg(not(feature = "metal4"))]
    {
        let _ = device;
        false
    }
}

#[cfg(feature = "metal")]
fn metal4_mtltensor_supported(device: &MetalDevice) -> bool {
    #[cfg(all(feature = "metal4", target_os = "macos", target_arch = "aarch64"))]
    {
        let _ = device;
        static PROBE: OnceLock<bool> = OnceLock::new();
        *PROBE.get_or_init(|| {
            use objc2_metal::{
                MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
                MTLTensorDataType, MTLTensorDescriptor, MTLTensorUsage,
            };

            let Some(dev) = MTLCreateSystemDefaultDevice() else {
                return false;
            };

            let descriptor = MTLTensorDescriptor::new();
            descriptor.setDataType(MTLTensorDataType::Float32);
            descriptor.setUsage(MTLTensorUsage::Compute);

            let Some(buffer) = dev.newBufferWithLength_options(4, MTLResourceOptions::StorageModeShared)
            else {
                return false;
            };

            // Safety: zero offset with a tiny shared buffer and default descriptor.
            unsafe { buffer.newTensorWithDescriptor_offset_error(&descriptor, 0).is_ok() }
        })
    }
    #[cfg(not(all(feature = "metal4", target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = device;
        false
    }
}

#[cfg(feature = "metal")]
pub fn select_metal_backend(device: &MetalDevice) -> Result<MetalBackend> {
    let policy = std::env::var("CANDLE_VLLM_METAL_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    let metal4_supported = metal4_is_supported(device);
    let metal4_tensor_supported = metal4_mtltensor_supported(device);

    #[cfg(feature = "metal4-debug-asserts")]
    if metal4_supported && !metal4_tensor_supported {
        candle_core::bail!(
            "Metal4 build detected but MTLTensor probe failed; disable `metal4-debug-asserts` or use metal3 fallback"
        );
    }

    match policy.as_str() {
        "metal3" => Ok(MetalBackend::Metal3),
        "metal4" => {
            if metal4_supported {
                if std::env::var("CANDLE_VLLM_METAL4_REQUIRE_MTLTENSOR")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false)
                    && !metal4_tensor_supported
                {
                    candle_core::bail!(
                        "CANDLE_VLLM_METAL4_REQUIRE_MTLTENSOR is set but MTLTensor is unavailable on this runtime"
                    )
                }
                Ok(MetalBackend::Metal4)
            } else {
                candle_core::bail!(
                    "CANDLE_VLLM_METAL_BACKEND=metal4 requested, but Metal4 path is unavailable on this device/build"
                )
            }
        }
        "auto" => {
            if metal4_supported {
                Ok(MetalBackend::Metal4)
            } else {
                Ok(MetalBackend::Metal3)
            }
        }
        other => candle_core::bail!(
            "Invalid CANDLE_VLLM_METAL_BACKEND value '{other}', expected one of: metal3, metal4, auto"
        ),
    }
}
