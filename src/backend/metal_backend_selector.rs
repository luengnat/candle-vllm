#[cfg(feature = "metal")]
use candle_core::{MetalDevice, Result};

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
pub fn select_metal_backend(device: &MetalDevice) -> Result<MetalBackend> {
    let policy = std::env::var("CANDLE_VLLM_METAL_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase();
    let metal4_supported = metal4_is_supported(device);

    match policy.as_str() {
        "metal3" => Ok(MetalBackend::Metal3),
        "metal4" => {
            if metal4_supported {
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
