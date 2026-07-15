use std::sync::{Mutex, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{Status, catch_panic, set_last_error};

static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();
static INIT_LOCK: Mutex<()> = Mutex::new(());

fn build_thread_pool(count: usize) -> Result<ThreadPool, Status> {
    let mut builder = ThreadPoolBuilder::new();
    if count != 0 {
        builder = builder.num_threads(count);
    }
    builder.build().map_err(|e| {
        set_last_error(format!("failed to create compression thread pool: {e}"));
        Status::Internal
    })
}

fn thread_pool() -> Result<&'static ThreadPool, Status> {
    if let Some(pool) = THREAD_POOL.get() {
        return Ok(pool);
    }

    let _guard = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pool) = THREAD_POOL.get() {
        return Ok(pool);
    }

    let pool = build_thread_pool(0)?;
    Ok(THREAD_POOL.get_or_init(|| pool))
}

/// Configure the process-wide compression worker pool used by the C API.
///
/// A `count` of zero selects Rayon's platform default. Positive values
/// request exactly that many workers; one worker is effectively serial. The
/// pool is created once per process, so this function may succeed at most
/// once and must be called before the first `ctt_convert`; afterwards it
/// returns `CTT_STATUS_THREAD_POOL_ALREADY_INITIALIZED`.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_set_thread_count(count: usize) -> Status {
    catch_panic(Status::Internal, || {
        let _guard = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        if THREAD_POOL.get().is_some() {
            set_last_error(
                "the compression thread pool is already initialized; ctt_set_thread_count may \
                 succeed at most once and must be called before the first conversion",
            );
            return Status::ThreadPoolAlreadyInitialized;
        }

        let pool = match build_thread_pool(count) {
            Ok(pool) => pool,
            Err(status) => return status,
        };
        THREAD_POOL.get_or_init(|| pool);
        Status::Ok
    })
}

pub(crate) fn install<T: Send>(f: impl FnOnce() -> T + Send) -> Result<T, Status> {
    Ok(thread_pool()?.install(f))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn concurrent_installs_share_the_configured_pool() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();
        assert_eq!(rayon::current_num_threads(), 1);
        assert_eq!(ctt_set_thread_count(2), Status::Ok);

        let barrier = Arc::new(Barrier::new(3));
        let callers: Vec<_> = (0..2)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    install(|| {
                        rayon::broadcast(|_| std::thread::current().id())
                            .into_iter()
                            .collect::<HashSet<_>>()
                    })
                    .unwrap()
                })
            })
            .collect();
        barrier.wait();

        let worker_sets: Vec<_> = callers
            .into_iter()
            .map(|caller| caller.join().unwrap())
            .collect();
        assert_eq!(worker_sets[0].len(), 2);
        assert_eq!(worker_sets[0], worker_sets[1]);
        assert_eq!(rayon::current_num_threads(), 1);
    }
}
