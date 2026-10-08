use pcg_rand::Pcg64;
use pi_arr::{arr, Arr, VBArr, VecArr};
use rand::{Rng, SeedableRng};
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[test]
fn macros_access_clone_and_shared_mutation() {
    let mut a: Arr<usize> = arr![1; 3];
    assert_eq!(a.slice(0..3).copied().collect::<Vec<_>>(), [1, 1, 1]);
    assert_eq!(a.set(1, 40), 1);
    assert_eq!(a.get_mut(1), Some(&mut 40));
    unsafe {
        assert_eq!(*a.get_unchecked(1), 40);
        *a.get_unchecked_mut(1) = 20;
        *a.load(1).unwrap() = 30;
        *a.load_unchecked(1) = 40;
        assert_eq!(*a.load_alloc(1), 40);
        a.insert(133, 97);
    }
    assert_eq!(a[133], 97);
    let b = a.clone();
    assert_eq!(b[1], 40);
    assert_eq!(b[133], 97);
    assert!(a.get(usize::MAX).is_none());
}
#[test]
fn positive_capacity_initializes_strings_and_atomics() {
    let a = VBArr::<String>::with_capacity(7);
    assert_eq!(a.vec_capacity(), 7);
    for i in 0..7 {
        assert_eq!(a[i], "");
    }
    assert!(a.get(7).is_none());
    let b = VecArr::<String>::with_capacity(7);
    for i in 0..7 {
        assert_eq!(b[i], "");
    }
    let a = VBArr::<AtomicUsize>::with_capacity(3);
    a[1].store(9, Ordering::Relaxed);
    assert_eq!(a[1].load(Ordering::Relaxed), 9);
}

#[test]
fn disjoint_shared_writes_do_not_borrow_whole_prefix() {
    let a = VBArr::<usize>::with_capacity(4);
    let reader = a.get(0).unwrap();
    unsafe {
        *a.load(1).unwrap() = 7;
    }
    assert_eq!(*reader, 0);
    assert_eq!(a[1], 7);
    let b = VecArr::<usize>::with_capacity(4);
    let reader = b.get(0).unwrap();
    unsafe {
        *b.load_alloc(1) = 9;
    }
    assert_eq!(*reader, 0);
    assert_eq!(b[1], 9);
}
#[test]
fn prefix_bucket_iterator_segments_and_boundaries() {
    let mut a = VBArr::<usize>::with_capacity(3);
    a.extend([1, 2, 4]);
    for i in [3, 34, 35, 98, 99, 226] {
        a.set(i, i);
    }
    assert_eq!(a.capacity(3), 3);
    assert_eq!(a.capacity(4), 35);
    assert_eq!(a.capacity(35), 35);
    assert_eq!(a.capacity(36), 99);
    assert_eq!(a.slice(1..3).copied().collect::<Vec<_>>(), [2, 4]);
    let parts = a.segments(1..101).collect::<Vec<_>>();
    assert_eq!(
        parts.iter().map(|s| s.len()).collect::<Vec<_>>(),
        [2, 32, 64, 2]
    );
    assert_eq!(parts.concat(), a.slice(1..101).copied().collect::<Vec<_>>());
    let mut it = a.slice(32..38);
    for index in 32..38 {
        assert_eq!(it.next(), a.get(index));
        assert_eq!(it.index(), index + 1);
    }
    assert!(it.next().is_none());
    assert_eq!(a.slice(10..10).count(), 0);
    assert_eq!(a.get(227), None);
}
#[test]
fn retain_across_unallocated_gaps_fills_defaults() {
    let mut a = VBArr::<String>::with_capacity(5);
    a.set(2, "prefix".into());
    a.set(105, "bucket".into());
    a.remain_settle(2..110, 110, 4);
    assert_eq!(a[0], "prefix");
    assert_eq!(a[103], "bucket");
    for i in 1..103 {
        assert_eq!(a[i], "");
    }
    assert_eq!(a.vec_capacity(), 112);
    a.clear(1, 0);
    assert_eq!(a.vec_capacity(), 112);
    for i in 0..112 {
        assert_eq!(a[i], "");
    }
    assert!(a.get(112).is_none());
}

#[test]
fn settle_defaults_tail_equally_in_both_arrays() {
    let mut buckets = VBArr::<String>::with_capacity(4);
    let mut contiguous = VecArr::<String>::with_capacity(4);
    buckets.set(0, "retained".into());
    contiguous.set(0, "retained".into());
    buckets.set(3, "discarded".into());
    contiguous.set(3, "discarded".into());
    buckets.settle(1, 0);
    contiguous.settle(1, 0);
    assert_eq!(buckets.vec_capacity(), 4);
    assert_eq!(contiguous.vec_capacity(), 4);
    assert_eq!(
        contiguous.slice(0..4).cloned().collect::<Vec<_>>(),
        ["retained", "", "", ""]
    );
    assert_eq!(
        buckets.slice(0..4).collect::<Vec<_>>(),
        contiguous.slice(0..4).collect::<Vec<_>>()
    );
}

#[test]
fn settle_beyond_old_capacity_defaults_gaps_equally() {
    let mut buckets = VBArr::<String>::with_capacity(4);
    let mut contiguous = VecArr::<String>::with_capacity(4);
    buckets.set(3, "retained".into());
    contiguous.set(3, "retained".into());
    buckets.settle(7, 2);
    contiguous.settle(7, 2);
    assert_eq!(buckets.vec_capacity(), 9);
    assert_eq!(contiguous.vec_capacity(), 9);
    assert_eq!(contiguous[3], "retained");
    assert!(contiguous.slice(4..9).all(String::is_empty));
    assert_eq!(
        buckets.slice(0..9).collect::<Vec<_>>(),
        contiguous.slice(0..9).collect::<Vec<_>>()
    );
}
#[test]
fn randomized_sparse_retain_matches_vec() {
    // 迁移原 test22 的确定性随机保留测试，标准 Vec 替代 VecRemain。
    let mut rng = Pcg64::seed_from_u64(300000);
    let mut a = Arr::<usize>::new();
    let mut expected = Vec::new();
    for _ in 0..20 {
        a.clear(expected.len(), 0);
        expected.clear();
        for _ in 0..20 {
            let index = rng.gen_range(1..300);
            a.set(index, index);
            expected.resize(expected.len().max(index + 1), 0);
            expected[index] = index;
        }
        let end = rng.gen_range(expected.len() / 2..expected.len());
        let start = rng.gen_range(0..end);
        a.remain_settle(start..end, expected.len(), 7);
        expected = expected[start..end].to_vec();
        assert_eq!(
            a.slice(0..expected.len()).copied().collect::<Vec<_>>(),
            expected
        );
    }
}
#[test]
fn sequential_settle_matches_vec() {
    // 迁移原 test2 的顺序插入、整理、范围保留与 clear 覆盖。
    let mut a = Arr::<usize>::new();
    for round in 0..10 {
        for index in 0..100 {
            unsafe {
                a.insert(index, index);
            }
        }
        a.settle(100, round);
        assert_eq!(
            a.slice(0..100).copied().collect::<Vec<_>>(),
            (0..100).collect::<Vec<_>>()
        );
        a.remain_settle(11..90, 100, 3);
        assert_eq!(
            a.slice(0..79).copied().collect::<Vec<_>>(),
            (11..90).collect::<Vec<_>>()
        );
        a.clear(79, 0);
        assert!(a.slice(0..79).all(|v| *v == 0));
    }
}
#[test]
fn clone_preserves_arc_ownership() {
    let mut a = Arr::<Option<Arc<usize>>>::new();
    for i in 0..6 {
        a.set(i, Some(Arc::new(i)));
    }
    let b = a.clone();
    assert_eq!(Arc::strong_count(a[0].as_ref().unwrap()), 2);
    drop(a);
    assert_eq!(Arc::strong_count(b[0].as_ref().unwrap()), 1);
}
#[test]
fn concurrent_disjoint_raw_allocation_and_shared_reads() {
    let a = Arc::new(VBArr::<AtomicUsize>::with_capacity(3));
    let reader = a.get(0).unwrap();
    std::thread::scope(|scope| {
        for index in [1, 3, 34, 35, 99, 100] {
            let a = &a;
            scope.spawn(move || {
                let ptr = a.load_alloc_ptr(index);
                // 仅使用 AtomicUsize 内部可变性，共享读取不与 &mut 别名重叠。
                unsafe { &*ptr }.store(index, Ordering::Relaxed);
                assert_eq!(a[index].load(Ordering::Relaxed), index);
            });
        }
        assert_eq!(reader.load(Ordering::Relaxed), 0);
    });
    for index in [1, 3, 34, 35, 99, 100] {
        assert_eq!(a[index].load(Ordering::Relaxed), index);
    }
}
#[test]
fn shared_mutex_elements_are_synchronized() {
    let a = Arc::new(VBArr::<Mutex<usize>>::new());
    a.load_alloc_ptr(0);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            *a[0].lock().unwrap() += 1;
        });
    });
    assert_eq!(*a[0].lock().unwrap(), 1);
}
#[test]
fn vecarr_shared_growth_then_reads_and_settle() {
    let a = VecArr::<String>::with_capacity(2);
    unsafe {
        a.insert(65, "value".into());
    }
    assert_eq!(a[65], "value");
    assert_eq!(a.slice(64..100).count(), 2);
    assert_eq!(a.segments(64..100).next().unwrap().len(), 2);
    let mut a = a;
    a.remain_settle(64..66, 66, 0);
    assert_eq!(a[1], "value");
    a.clear(1, 0);
    assert_eq!(a[1], "");
    a.settle(80, 3);
    assert_eq!(a.vec_capacity(), 83);
}

thread_local! {
    static MADE: Cell<usize> = const { Cell::new(0) };
    static DROPPED: Cell<usize> = const { Cell::new(0) };
    static DEFAULT_LEFT: Cell<usize> = const { Cell::new(usize::MAX) };
    static CLONE_LEFT: Cell<usize> = const { Cell::new(usize::MAX) };
}
fn reset() {
    MADE.set(0);
    DROPPED.set(0);
    DEFAULT_LEFT.set(usize::MAX);
    CLONE_LEFT.set(usize::MAX);
}
fn consume(counter: &'static std::thread::LocalKey<Cell<usize>>) {
    let left = counter.get();
    assert_ne!(left, 0, "injected panic");
    counter.set(left - 1);
}
#[derive(Debug)]
struct Tracked(String);
impl Default for Tracked {
    fn default() -> Self {
        consume(&DEFAULT_LEFT);
        MADE.set(MADE.get() + 1);
        Self("default".into())
    }
}
impl Clone for Tracked {
    fn clone(&self) -> Self {
        consume(&CLONE_LEFT);
        MADE.set(MADE.get() + 1);
        Self(self.0.clone())
    }
}
impl Drop for Tracked {
    fn drop(&mut self) {
        DROPPED.set(DROPPED.get() + 1);
    }
}
#[test]
fn default_panics_reclaim_initialization_and_bucket_ownership() {
    reset();
    DEFAULT_LEFT.set(2);
    assert!(catch_unwind(|| VBArr::<Tracked>::with_capacity(4)).is_err());
    assert_eq!(MADE.get(), DROPPED.get());
    reset();
    let a = VBArr::<Tracked>::with_capacity(3);
    DEFAULT_LEFT.set(2);
    assert!(catch_unwind(AssertUnwindSafe(|| a.load_alloc_ptr(4))).is_err());
    assert_eq!(a[0].0, "default");
    DEFAULT_LEFT.set(usize::MAX);
    a.load_alloc_ptr(4);
    drop(a);
    assert_eq!(MADE.get(), DROPPED.get());
}
#[test]
fn clone_panics_keep_source_owned_and_reclaim_partial_clones() {
    reset();
    let mut a = VBArr::<Tracked>::with_capacity(3);
    a.alloc(5);
    for count in [1, 4] {
        CLONE_LEFT.set(count);
        assert!(catch_unwind(AssertUnwindSafe(|| a.clone())).is_err());
        assert_eq!(a[0].0, "default");
        assert_eq!(a[5].0, "default");
    }
    drop(a);
    assert_eq!(MADE.get(), DROPPED.get());
    reset();
    let a = VecArr::<Tracked>::with_capacity(3);
    CLONE_LEFT.set(1);
    assert!(catch_unwind(AssertUnwindSafe(|| a.clone())).is_err());
    assert_eq!(a[0].0, "default");
    drop(a);
    assert_eq!(MADE.get(), DROPPED.get());
}
#[test]
fn settle_and_clear_default_panics_leave_valid_owners() {
    reset();
    let mut a = VBArr::<Tracked>::with_capacity(2);
    a.alloc(100);
    DEFAULT_LEFT.set(1);
    assert!(catch_unwind(AssertUnwindSafe(|| a.remain_settle(1..105, 105, 3))).is_err());
    assert!(a.get(0).is_none());
    DEFAULT_LEFT.set(usize::MAX);
    a.settle(3, 0);
    DEFAULT_LEFT.set(1);
    assert!(catch_unwind(AssertUnwindSafe(|| a.clear(5, 0))).is_err());
    assert!(a.get(0).is_none());
    drop(a);
    assert_eq!(MADE.get(), DROPPED.get());
}
#[test]
fn vec_growth_default_panic_refreshes_or_empties_cache() {
    reset();
    let mut a = VecArr::<Tracked>::with_capacity(2);
    DEFAULT_LEFT.set(1);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        a.alloc(7);
    }))
    .is_err());
    assert_eq!(a.vec_capacity(), 3);
    assert_eq!(a[2].0, "default");
    DEFAULT_LEFT.set(1);
    assert!(catch_unwind(AssertUnwindSafe(|| unsafe {
        a.load_alloc_ptr(9);
    }))
    .is_err());
    assert!(a.get(0).is_none());
    DEFAULT_LEFT.set(usize::MAX);
    a.alloc(0);
    DEFAULT_LEFT.set(1);
    assert!(catch_unwind(AssertUnwindSafe(|| a.clear(4, 0))).is_err());
    assert_eq!(a[0].0, "default");
    DEFAULT_LEFT.set(usize::MAX);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        DEFAULT_LEFT.set(0);
        a.remain_settle(0..2, 2, 3);
    }))
    .is_err());
    assert!(a.get(0).is_none());
    drop(a);
    assert_eq!(MADE.get(), DROPPED.get());
}
struct Zst;

thread_local! {
    static DROP_PANIC: Cell<usize> = const { Cell::new(usize::MAX) };
}
struct PanicDrop(usize);
impl Default for PanicDrop {
    fn default() -> Self {
        MADE.set(MADE.get() + 1);
        Self(usize::MAX)
    }
}
impl Drop for PanicDrop {
    fn drop(&mut self) {
        DROPPED.set(DROPPED.get() + 1);
        if self.0 == DROP_PANIC.get() && self.0 != usize::MAX {
            DROP_PANIC.set(usize::MAX);
            panic!("injected destructor panic");
        }
    }
}
#[test]
fn settle_drop_panics_leave_empty_valid_arrays_and_reclaim_values() {
    for (index, range) in [(0, 1..10), (7, 0..7), (8, 9..10), (9, 8..9)] {
        reset();
        let mut a = VBArr::<PanicDrop>::with_capacity(8);
        a.alloc(10);
        a.get_mut(index).unwrap().0 = index;
        DROP_PANIC.set(index);
        assert!(catch_unwind(AssertUnwindSafe(|| a.remain_settle(range, 11, 0))).is_err());
        assert_eq!(a.vec_capacity(), 0);
        assert!(a.get(0).is_none());
        a.alloc(0);
        drop(a);
        assert_eq!(MADE.get(), DROPPED.get());
        reset();
        let mut b = VecArr::<PanicDrop>::with_capacity(11);
        b.get_mut(index).unwrap().0 = index;
        DROP_PANIC.set(index);
        // Exclude each selected value so its destructor runs during settling.
        assert!(catch_unwind(AssertUnwindSafe(|| b.remain_settle(index + 1..11, 11, 0))).is_err());
        assert_eq!(b.vec_capacity(), 0);
        assert!(b.get(0).is_none());
        b.alloc(0);
        drop(b);
        assert_eq!(MADE.get(), DROPPED.get());
    }
}

#[test]
fn settle_reuses_main_allocations_and_moves_bucket_strings() {
    let mut a = VBArr::<String>::with_capacity(128);
    a.set(4, "prefix".into());
    a.set(130, "bucket".into());
    let p = a.get(0).unwrap() as *const String;
    a.remain_settle(4..132, 132, 0);
    assert_eq!(a.get(0).unwrap() as *const String, p);
    assert_eq!(a[0], "prefix");
    assert_eq!(a[126], "bucket");
    assert!(a.get(128).is_none());
    let mut b = VecArr::<String>::with_capacity(128);
    b.set(4, "prefix".into());
    let p = b.get(0).unwrap() as *const String;
    b.remain_settle(4..8, 128, 0);
    assert_eq!(b.get(0).unwrap() as *const String, p);
    assert_eq!(b[0], "prefix");
    assert!(b.slice(1..128).all(String::is_empty));
}
impl Default for Zst {
    fn default() -> Self {
        MADE.set(MADE.get() + 1);
        Self
    }
}
impl Clone for Zst {
    fn clone(&self) -> Self {
        MADE.set(MADE.get() + 1);
        Self
    }
}
impl Drop for Zst {
    fn drop(&mut self) {
        DROPPED.set(DROPPED.get() + 1);
    }
}
#[test]
fn zst_defaults_and_destructors_match_initialized_slots() {
    reset();
    let mut a = VBArr::<Zst>::with_capacity(3);
    assert_eq!(MADE.get(), 3);
    a.alloc(3);
    assert_eq!(MADE.get(), 35);
    assert_eq!(a.slice(0..35).count(), 35);
    assert!(a.get(35).is_none());
    let b = a.clone();
    assert_eq!(MADE.get(), 70);
    drop(b);
    a.remain_settle(1..5, 35, 1);
    a.clear(5, 0);
    drop(a);
    assert_eq!(MADE.get(), DROPPED.get());
    reset();
    let mut a = VecArr::<Zst>::with_capacity(3);
    a.alloc(5);
    let b = a.clone();
    drop(b);
    a.remain_settle(2..6, 6, 1);
    a.clear(2, 0);
    drop(a);
    assert_eq!(MADE.get(), DROPPED.get());
}
