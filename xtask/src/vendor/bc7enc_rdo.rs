use std::path::Path;

use anyhow::Result;

use super::{
    clean_and_create, copy_text_file, read_text, replace_required, require_dir, write_text,
};
use crate::util::workspace_root;

/// Automatically vendored from <https://github.com/richgel999/bc7enc_rdo>.
/// Regenerate with: `cargo xtask vendor bc7enc-rdo [--src <path>]`
pub fn vendor_bc7enc_rdo(src_dir: &Path) -> Result<()> {
    let ws = workspace_root();
    require_dir(src_dir)?;

    let dst_dir = ws.join("crates/ctt-bc7enc-rdo/ispc");
    clean_and_create(&dst_dir)?;

    let dst = dst_dir.join("bc7e.ispc");
    copy_text_file(&src_dir.join("bc7e.ispc"), &dst)?;
    patch_bc7e_ispc(&dst)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// bc7enc_rdo patches
// ---------------------------------------------------------------------------

/// Local fix carried on top of upstream `bc7e.ispc` (see the "divergent bc7enc
/// ISPC bit packing" change): pack the BC7 block through two `uint64` halves
/// instead of a per-byte OR loop, and avoid the `--` operator on unsigned
/// varying values (ISPC silently drops it). The two changes travel together
/// because the packing rewrite is what makes the encoder correct under ISPC's
/// varying execution.
const DECREMENT_COMMENT: &str = " // ISPC drops `--` on unsigned; use `-= 1`";

fn patch_bc7e_ispc(path: &Path) -> Result<()> {
    let mut text = read_text(path)?;

    // Rewrite set_block_bits to accumulate into two uint64 halves.
    replace_required(
        &mut text,
        concat!(
            "static inline void set_block_bits(uint8_t *pBytes, uint32_t val, uint32_t num_bits, varying uint32_t *uniform pCur_ofs)\n",
            "{\n",
            "\tassert(num_bits < 32);\n",
            "\tuint32_t limit = 1U << num_bits;\n",
            "\tassert(val < limit);\n",
            "\t\t\n",
            "\twhile (num_bits)\n",
            "\t{\n",
            "\t\tconst uint32_t n = minimumu(8 - (*pCur_ofs & 7), num_bits);\n",
            "\n",
            "#pragma ignore warning(perf)\n",
            "\t\tpBytes[*pCur_ofs >> 3] |= (uint8_t)(val << (*pCur_ofs & 7));\n",
            "\n",
            "\t\tval >>= n;\n",
            "\t\tnum_bits -= n;\n",
            "\t\t*pCur_ofs += n;\n",
            "\t}\n",
            "\n",
            "\tassert(*pCur_ofs <= 128);\n",
            "}",
        ),
        concat!(
            "static inline void set_block_bits(varying uint64_t *uniform pLow, varying uint64_t *uniform pHigh, uint32_t val, uint32_t num_bits, varying uint32_t *uniform pCur_ofs)\n",
            "{\n",
            "\tassert(num_bits < 32);\n",
            "\tuint32_t limit = 1U << num_bits;\n",
            "\tassert(val < limit);\n",
            "\n",
            "\tconst uint32_t cur_ofs = *pCur_ofs;\n",
            "\tif (cur_ofs < 64)\n",
            "\t{\n",
            "\t\t*pLow |= (uint64_t)val << cur_ofs;\n",
            "\t\tif (cur_ofs + num_bits > 64)\n",
            "\t\t\t*pHigh |= (uint64_t)val >> (64 - cur_ofs);\n",
            "\t}\n",
            "\telse\n",
            "\t\t*pHigh |= (uint64_t)val << (cur_ofs - 64);\n",
            "\n",
            "\t*pCur_ofs += num_bits;\n",
            "\n",
            "\tassert(*pCur_ofs <= 128);\n",
            "}",
        ),
        "bc7e.ispc: set_block_bits uint64 packing",
    )?;

    // Accumulate into two uint64 halves instead of zeroing the block up front.
    replace_required(
        &mut text,
        concat!(
            "\tuint8_t *pBlock_bytes = (uint8_t *)(pBlock);\n",
            "\tmemset(pBlock_bytes, 0, BC7E_BLOCK_SIZE);",
        ),
        concat!("\tuint64_t block_low = 0;\n", "\tuint64_t block_high = 0;",),
        "bc7e.ispc: block accumulator declarations",
    )?;

    // Route every emitter through the two-half accumulator.
    replace_required(
        &mut text,
        "set_block_bits(pBlock_bytes, ",
        "set_block_bits(&block_low, &block_high, ",
        "bc7e.ispc: set_block_bits call sites",
    )?;

    // Store the accumulated halves once the block is complete.
    replace_required(
        &mut text,
        concat!(
            "\tassert(cur_bit_ofs == 128);\n",
            "}\n",
            "\n",
            "static inline void encode_bc7_block_mode6",
        ),
        concat!(
            "\tassert(cur_bit_ofs == 128);\n",
            "\n",
            "#pragma ignore warning(perf)\n",
            "\t((uint64_t *)(pBlock))[0] = block_low;\n",
            "\n",
            "#pragma ignore warning(perf)\n",
            "\t((uint64_t *)(pBlock))[1] = block_high;\n",
            "}\n",
            "\n",
            "static inline void encode_bc7_block_mode6",
        ),
        "bc7e.ispc: store accumulated halves",
    )?;

    // Replace the `--` decrements the packing path relies on. Each token is
    // present only in the varying-selector paths this fix touches.
    for (token, label) in [("sel--;", "sel--"), ("n--;", "n--"), ("la--;", "la--")] {
        let replacement = format!("{}{DECREMENT_COMMENT}", token.replace("--;", " -= 1;"));
        replace_required(
            &mut text,
            token,
            &replacement,
            &format!("bc7e.ispc: {label}"),
        )?;
    }

    write_text(path, &text)
}
