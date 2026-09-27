//! Input preparation shared by the BC6H encoders.

/// Copy f16 data and replace each negative value with +0.
///
/// The BC6H UFLOAT encoders read f16 bits as unsigned integers, so a value
/// with the sign bit set saturates to the f16 maximum.
pub(crate) fn clamp_negative_f16(data: &[u8]) -> Vec<u16> {
    let (halves, _) = data.as_chunks::<2>();
    halves
        .iter()
        .map(|&bytes| {
            let h = u16::from_ne_bytes(bytes);
            if h & 0x8000 != 0 { 0 } else { h }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn negatives_become_zero() {
        let input = [-0.04, -0.0, 0.0, 0.5, 65504.0, -65504.0].map(f16::from_f32);
        let clamped = clamp_negative_f16(bytemuck::cast_slice(&input));
        let expected = [0.0, 0.0, 0.0, 0.5, 65504.0, 0.0].map(|v| f16::from_f32(v).to_bits());
        assert_eq!(clamped, expected);
    }
}
