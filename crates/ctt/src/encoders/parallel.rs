//! Feature-gated helpers for splitting encoded output into block-row chunks.

/// Infallible form of [`try_for_each_row_chunk`].
#[cfg(any(
    feature = "encoder-intel",
    feature = "encoder-bc7enc",
    feature = "encoder-etcpak",
))]
pub(crate) fn for_each_row_chunk<T: Send>(
    output: &mut [T],
    row_len: usize,
    f: impl Fn(usize, usize, &mut [T]) + Send + Sync,
) {
    try_for_each_row_chunk(output, row_len, |start, count, chunk| {
        f(start, count, chunk);
        Ok::<(), std::convert::Infallible>(())
    })
    .unwrap();
}

/// Run `f` over disjoint groups of output block rows.
///
/// Without the `rayon` feature this always makes one call covering the entire
/// output. Parallel builds target four jobs per active worker, which gives
/// Rayon enough work to balance slow blocks without creating one task per row.
pub(crate) fn try_for_each_row_chunk<T: Send, E: Send>(
    output: &mut [T],
    row_len: usize,
    f: impl Fn(usize, usize, &mut [T]) -> Result<(), E> + Send + Sync,
) -> Result<(), E> {
    assert!(row_len > 0, "encoded block rows must be non-empty");
    assert_eq!(
        output.len() % row_len,
        0,
        "encoded output must contain whole block rows"
    );
    let rows = output.len() / row_len;
    assert!(
        rows > 0,
        "encoded output must contain at least one block row"
    );

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;

        let workers = rayon::current_num_threads();
        let rows_per_chunk = rows.div_ceil(workers * 4);
        if workers > 1 && rows_per_chunk < rows {
            return output
                .par_chunks_mut(rows_per_chunk * row_len)
                .enumerate()
                .try_for_each(|(chunk_index, chunk)| {
                    let start = chunk_index * rows_per_chunk;
                    f(start, chunk.len() / row_len, chunk)
                });
        }
    }

    f(0, rows, output)
}

#[cfg(all(test, feature = "rayon"))]
mod tests {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex};

    #[test]
    fn one_surface_uses_multiple_workers() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let workers = Mutex::new(HashSet::new());
        let rendezvous = Barrier::new(2);
        let mut output = vec![0u8; 256 * 16];

        // The barrier below releases jobs in pairs, which deadlocks instead of
        // failing if the chunking heuristic ever produces an odd job count, so
        // check the count up front.
        let jobs = AtomicUsize::new(0);
        pool.install(|| {
            super::try_for_each_row_chunk(&mut output, 16, |_, _, _| {
                jobs.fetch_add(1, Ordering::Relaxed);
                Ok::<(), std::convert::Infallible>(())
            })
            .unwrap();
        });
        let jobs = jobs.into_inner();
        assert!(jobs > 1 && jobs.is_multiple_of(2), "got {jobs} jobs");

        pool.install(|| {
            super::try_for_each_row_chunk(&mut output, 16, |_, _, chunk| {
                workers.lock().unwrap().insert(std::thread::current().id());
                // Pair jobs so both workers must participate before either can
                // consume the remaining work.
                rendezvous.wait();
                chunk.fill(1);
                Ok::<(), std::convert::Infallible>(())
            })
            .unwrap();
        });

        assert!(workers.into_inner().unwrap().len() > 1);
        assert!(output.iter().all(|&byte| byte == 1));
    }
}
