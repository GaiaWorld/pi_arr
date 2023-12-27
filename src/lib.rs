//! 线程安全的数组
//! 迭代性能比Vec慢1-10倍， 主要损失在切换bucket时，原子操作及缓存失效。

#![feature(vec_into_raw_parts)]
#![feature(const_option)]
#![feature(test)]
extern crate test;

use std::mem::{forget, replace, transmute, MaybeUninit};
use std::ops::{Index, IndexMut, Range};
use std::sync::atomic::Ordering;
use std::{fmt, ptr};

use pi_share::{ShareMutex, SharePtr};

const BUCKETS: usize = (u32::BITS as usize) - SKIP_BUCKET;
const MAX_ENTRIES: usize = (u32::MAX as usize) - SKIP;

/// Creates a [`Arr`] containing the given elements.
///
/// `arr!` allows `Arr`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`Arr`] containing a given list of elements:
///
/// ```
/// let arr = pi_arr::arr![1, 2, 3];
/// unsafe{
///     assert_eq!(*arr[0].as_ptr(), 1);
///     assert_eq!(*arr[1].as_ptr(), 2);
///     assert_eq!(*arr[2].as_ptr(), 3);
/// }
/// ```
///
/// - Create a [`Arr`] from a given element and size:
///
/// ```
/// let arr = pi_arr::arr![1; 3];
/// unsafe{
///     assert_eq!(*arr[0].as_ptr(), 1);
///     assert_eq!(*arr[1].as_ptr(), 1);
///     assert_eq!(*arr[2].as_ptr(), 1);
/// }
/// ```
#[macro_export]
macro_rules! arr {
    () => {
        $crate::Arr::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut arr = $crate::Arr::with_capacity($n);
        arr.extend(::core::iter::repeat($elem).take($n));
        arr
    }};
    ($($x:expr),+ $(,)?) => (
        <$crate::Arr<_> as core::iter::FromIterator<_>>::from_iter([$($x),+])
    );
}

/// A lock-free, auto-expansion array.
///
/// See [the crate documentation](crate) for details.
///
/// # Notes
///
/// The bucket array is stored inline, meaning that the
/// `Arr<T>` is quite large. It is expected that you
/// store it behind an [`Arc`](std::sync::Arc) or similar.
#[derive(Default)]
pub struct Arr<T> {
    buckets: [Bucket<T>; BUCKETS],
    lock: ShareMutex<()>,
}

unsafe impl<T: Send> Send for Arr<T> {}
unsafe impl<T: Sync> Sync for Arr<T> {}

impl<T> Arr<T> {
    /// Constructs a new, empty `Arr<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr: pi_arr::Arr<i32> = pi_arr::Arr::new();
    /// ```
    #[inline]
    pub fn new() -> Arr<T> {
        Arr::with_capacity(0)
    }

    /// Constructs a new, empty `Arr<T>` with the specified capacity.
    ///
    /// Capacity will be aligned to a power of 2 size.
    /// The array will be able to hold at least `capacity` elements
    /// without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut arr = pi_arr::Arr::with_capacity(32);
    ///
    /// for i in 0..32 {
    ///     // will not allocate
    ///     arr.alloc(&pi_arr::Location::of(i)).write(i);
    /// }
    ///
    /// // may allocate
    /// arr.alloc(&pi_arr::Location::of(32)).write(32);
    /// ```
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Arr<T> {
        let mut buckets = [ptr::null_mut(); BUCKETS];
        if capacity == 0 {
            return Arr {
                buckets: buckets.map(Bucket::new),
                lock: ShareMutex::default(),
            };
        }
        let end = Location::of(capacity).bucket;
        for (i, bucket) in buckets[..=end].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = Bucket::alloc(len);
        }

        Arr {
            buckets: buckets.map(Bucket::new),
            lock: ShareMutex::default(),
        }
    }

    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// unsafe{
    /// assert_eq!(40, *arr.get(&pi_arr::Location::of(1)).unwrap().as_ptr());
    /// assert_eq!(true, arr.get(&pi_arr::Location::of(33)).is_none());
    /// }
    /// ```
    #[inline(always)]
    pub fn get(&self, location: &Location) -> Option<&MaybeUninit<T>> {
        // safety: `location.bucket` is always in bounds
        let entries = unsafe { self.entries(location.bucket) };

        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }

        // safety: `location.entry` is always in bounds for it's bucket
        Some(unsafe { &*entries.add(location.entry) })
    }

    /// Returns a reference to an element, without doing bounds
    /// checking or verifying that the element is fully initialized.
    ///
    /// For a safe alternative see [`get`](Arr::get).
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index, or for an element that
    /// is being concurrently initialized is **undefined behavior**, even if
    /// the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(*arr.get_unchecked(&pi_arr::Location::of(1)).as_ptr(), 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, location: &Location) -> &MaybeUninit<T> {
        &*self.entries(location.bucket).add(location.entry)
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// unsafe{
    /// assert_eq!(40, *arr.get_mut(&pi_arr::Location::of(1)).unwrap().as_ptr());
    /// assert_eq!(true, arr.get_mut(&pi_arr::Location::of(33)).is_none());
    /// }
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, location: &Location) -> Option<&mut MaybeUninit<T>> {
        let entries = unsafe { self.entries(location.bucket) };

        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }

        // safety: `location.entry` is always in bounds for it's bucket
        Some(unsafe { &mut *entries.add(location.entry) })
    }

    /// Returns a mutable reference to an element, without doing bounds
    /// checking or verifying that the element is fully initialized.
    ///
    /// For a safe alternative see [`get`](Arr::get).
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is **undefined
    /// behavior**, even if the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(*arr.get_unchecked_mut(&pi_arr::Location::of(1)).as_ptr(), 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, location: &Location) -> &mut MaybeUninit<T> {
        &mut *self.entries(location.bucket).add(location.entry)
    }
    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// unsafe {
    /// assert_eq!(40, *arr.alloc(&pi_arr::Location::of(1)).as_ptr());
    /// }
    /// ```
    #[inline(always)]
    pub fn alloc(&mut self, location: &Location) -> &mut MaybeUninit<T> {
        let bucket = unsafe { self.buckets.get_unchecked_mut(location.bucket) };
        // safety: `location.bucket` is always in bounds
        let mut entries = *bucket.entries.get_mut();

        // bucket is uninitialized
        if entries.is_null() {
            entries = bucket.init(location.len, &self.lock);
        }

        // safety: `location.entry` is always in bounds for it's bucket
        unsafe { &mut *entries.add(location.entry) }
    }
    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will not be allocated automatically, and the returned None.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// unsafe{
    /// // assert_eq!(10, *arr.load(&pi_arr::Location::of(0)).unwrap().as_ptr());
    /// // assert_eq!(40, *arr.load(&pi_arr::Location::of(1)).unwrap().as_ptr());
    /// assert_eq!(true, arr.load(&pi_arr::Location::of(33)).is_none());
    /// }
    /// ```
    #[inline(always)]
    pub fn load(&self, location: &Location) -> Option<&mut MaybeUninit<T>> {
        // safety: `location.bucket` is always in bounds
        let entries = unsafe { self.load_entries(location.bucket) };

        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }

        // safety: `location.entry` is always in bounds for it's bucket
        Some(unsafe { &mut *entries.add(location.entry) })
    }

    /// Returns a mutable reference to an element, without doing bounds
    /// checking or verifying that the element is fully initialized.
    ///
    /// For a safe alternative see [`get`](Arr::get).
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is **undefined
    /// behavior**, even if the resulting reference is not used.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(*arr.load_unchecked(&pi_arr::Location::of(1)).as_ptr(), 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn load_unchecked(&self, location: &Location) -> &mut MaybeUninit<T> {
        &mut *self.load_entries(location.bucket).add(location.entry)
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is null
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// unsafe{
    /// assert_eq!(40, *arr.load_alloc(&pi_arr::Location::of(1)).as_ptr());
    /// }
    /// ```
    #[inline(always)]
    pub fn load_alloc(&self, location: &Location) -> &mut MaybeUninit<T> {
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };
        // safety: `location.bucket` is always in bounds
        let mut entries = bucket.entries.load(Ordering::Relaxed);
        // bucket is uninitialized
        if entries.is_null() {
            entries = bucket.init(location.len, &self.lock);
        }
        // safety: `location.entry` is always in bounds for it's bucket
        unsafe { transmute(entries.add(location.entry)) }
    }

    // /// clear all elements
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// use pi_null::Null;
    // /// let mut arr = pi_arr::arr![1, 2];
    // /// unsafe {arr.clear()};
    // /// arr.set(2, 3);
    // /// assert_eq!(arr[0], i32::null());
    // /// assert_eq!(arr[1], i32::null());
    // /// assert_eq!(arr[2], 3);
    // /// ```
    // pub unsafe fn clear(&mut self) {
    //     for (i, bucket) in self.buckets.iter().enumerate() {
    //         let entries = bucket.entries.load(Ordering::Relaxed);
    //         if entries.is_null() {
    //             continue;
    //         }
    //         let len = Location::bucket_len(i);
    //         // safety: in clear
    //         let mut vec = unsafe { Vec::from_raw_parts(entries as *mut T, len, len) };
    //         vec.clear();
    //         forget(vec);
    //     }
    // }
    /// replace buckets.
    pub fn replace(&self) -> [Vec<MaybeUninit<T>>; BUCKETS] {
        let mut buckets = [0; BUCKETS].map(|_| Vec::new());
        for (i, bucket) in self.buckets.iter().enumerate() {
            *unsafe { buckets.get_unchecked_mut(i) } = bucket.to_vec(Location::bucket_len(i));
        }
        buckets
    }

    /// Returns an iterator over the array.
    ///
    /// Values are yielded in the form `(index, value)`. The array may
    /// have in-progress concurrent writes that create gaps, so `index`
    /// may not be strictly sequential.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2, 4];
    /// arr.load_alloc(&pi_arr::Location::of(98)).write(98);
    /// let mut iterator = arr.iter();
    /// assert_eq!(iterator.size_hint().0, 32);
    /// let r = iterator.next().unwrap();
    /// unsafe {
    /// assert_eq!((iterator.index() - 1, *r.as_ptr()), (0, 1));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r.as_ptr()), (1, 2));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r.as_ptr()), (2, 4));
    /// for i in 3..32 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1), (i));
    /// }
    /// for i in 96..98 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1), (i));
    /// }
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r.as_ptr()), (98, 98));
    /// }
    /// for i in 99..224 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1), (i));
    /// }
    /// assert_eq!(iterator.next().is_none(), true);
    /// assert_eq!(iterator.size_hint().0, 0);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.slice(0..MAX_ENTRIES)
    }

    /// Returns an iterator over the array at the given range.
    ///
    /// Values are yielded in the form `(index, Entry)`.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2, 4, 6];
    /// let mut iterator = arr.slice(1..3);
    /// let r = iterator.next().unwrap();
    /// unsafe{
    /// assert_eq!(*r.as_ptr(), 2);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r.as_ptr(), 4);
    /// }
    /// assert_eq!(iterator.next().is_none(), true);
    /// ```
    #[inline(always)]
    pub fn slice(&self, range: Range<usize>) -> Iter<'_, T> {
        Iter::new(&self.buckets, range)
    }
    #[inline(always)]
    pub unsafe fn load_entries(&self, bucket: usize) -> *mut MaybeUninit<T> {
        self.buckets
            .get_unchecked(bucket)
            .entries
            .load(Ordering::Relaxed)
    }
    #[inline(always)]
    pub unsafe fn entries(&self, bucket: usize) -> *mut MaybeUninit<T> {
        *self.buckets.get_unchecked(bucket).entries.as_ptr()
    }
    #[inline(always)]
    pub unsafe fn entries_mut(&mut self, bucket: usize) -> *mut MaybeUninit<T> {
        *self.buckets.get_unchecked_mut(bucket).entries.get_mut()
    }
}

/// An iterator over the elements of a [`Arr<T>`].
///
/// See [`Arr::iter`] for details.

pub struct Iter<'a, T> {
    buckets: &'a [Bucket<T>],
    start: Location,
    end: Location,
    ptr: *mut MaybeUninit<T>,
}
impl<'a, T> Iter<'a, T> {
    #[inline(always)]
    pub fn empty() -> Self {
        Iter {
            buckets: &[],
            start: Location::default(),
            end: Location::default(),
            ptr: ptr::null_mut(),
        }
    }
    fn new(buckets: &'a [Bucket<T>], range: Range<usize>) -> Self {
        let mut start = Location::of(range.start);
        let end = Location::of(range.end);
        if start.bucket == end.bucket {
            start.len = end.entry;
        } else if start.bucket > end.bucket {
            start.len = 0;
        }
        let ptr = unsafe {
            buckets
                .get_unchecked(start.bucket)
                .entries
                .load(Ordering::Relaxed)
        };
        if ptr.is_null() {
            start.len = start.entry;
        }
        Iter {
            buckets,
            start,
            end,
            ptr,
        }
    }
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.start.index()
    }
    #[inline(always)]
    pub(crate) fn get(&mut self) -> &'a mut MaybeUninit<T> {
        unsafe { transmute(self.ptr.add(self.start.entry)) }
    }
    #[inline]
    pub(crate) fn next_bucket(&mut self) -> Option<&'a mut MaybeUninit<T>> {
        loop {
            if self.start.bucket >= self.end.bucket {
                return None;
            }
            self.start.bucket += 1;
            self.ptr = unsafe {
                transmute(
                    self.buckets
                        .get_unchecked(self.start.bucket)
                        .entries
                        .load(Ordering::Relaxed),
                )
            };
            if !self.ptr.is_null() {
                if self.start.bucket == self.end.bucket {
                    self.start.len = self.end.entry;
                } else {
                    self.start.len = Location::bucket_len(self.start.bucket);
                }
                self.start.entry = 1;
                return Some(unsafe { transmute(self.ptr) });
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.start.bucket == self.end.bucket {
            let n = self.start.len.saturating_sub(self.start.entry);
            return (n, Some(n));
        } else if self.start.bucket > self.end.bucket {
            return (0, Some(0));
        }
        // 最小为起始槽的entry数量
        let min = self.start.len - self.start.entry;
        let c = self.end.bucket - self.start.bucket;
        // 中间槽的entry数量
        let n = self.start.len * (1 << c);
        (min, Some(min + n + self.end.entry))
    }
}
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a mut MaybeUninit<T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start.entry < self.start.len {
            let r = self.get();
            self.start.entry += 1;
            return Some(r);
        }
        self.next_bucket()
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size_hint()
    }
}

impl<T> Index<usize> for Arr<T> {
    type Output = MaybeUninit<T>;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(&Location::of(index))
            .expect("no element found at index {index}")
    }
}
impl<T> IndexMut<usize> for Arr<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(&Location::of(index))
            .expect("no element found at index_mut {index}")
    }
}

// impl<T> Drop for Arr<T> {
//     fn drop(&mut self) {
//         for (i, bucket) in self.buckets.iter_mut().enumerate() {
//             let entries = *bucket.entries.get_mut() as *mut T;
//             if entries.is_null() {
//                 continue;
//             }
//             let len = Location::bucket_len(i);
//             // safety: in drop
//             unsafe { drop(Vec::from_raw_parts(entries, len, len)) }
//         }
//     }
// }

impl<T> FromIterator<T> for Arr<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let mut arr = Arr::with_capacity(lower);
        for (i, value) in iter.enumerate() {
            arr.alloc(&Location::of(i)).write(value);
        }
        arr
    }
}

impl<T> Extend<T> for Arr<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        for (i, value) in iter.enumerate() {
            self.alloc(&Location::of(i)).write(value);
        }
    }
}

impl<T: Copy> Clone for Arr<T> {
    fn clone(&self) -> Arr<T> {
        let mut buckets: [*mut MaybeUninit<T>; BUCKETS] = [ptr::null_mut(); BUCKETS];

        for (i, bucket) in buckets.iter_mut().enumerate() {
            let entries = unsafe { self.load_entries(i) };
            // bucket is uninitialized
            if entries.is_null() {
                continue;
            }
            let len = Location::bucket_len(i);
            let vec = unsafe { Vec::from_raw_parts(entries, len, len) };
            *bucket = vec.clone().into_raw_parts().0;
            forget(vec);
        }
        Arr {
            buckets: buckets.map(Bucket::new),
            lock: ShareMutex::default(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Arr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// impl<T: PartialEq> PartialEq for Arr<T> {
//     fn eq(&self, other: &Self) -> bool {
//         if self.len() != other.len() {
//             return false;
//         }
//         let mut it = self.iter();
//         let mut other = other.iter();
//         while let Some(v) = it.next() {
//             if let Some(v2) = other.next() {
//                 if v != v2 {
//                     return false;
//                 }
//             } else {
//                 return false;
//             }
//         }
//         true
//     }
// }

// impl<A, T> PartialEq<A> for Arr<T>
// where
//     A: AsRef<[T]>,
//     T: PartialEq,
// {
//     fn eq(&self, other: &A) -> bool {
//         let other = other.as_ref();

//         if self.len() != other.len() {
//             return false;
//         }
//         let mut it = self.iter();
//         let mut other = other.iter();
//         while let Some(v) = it.next() {
//             if let Some(v2) = other.next() {
//                 if v != v2 {
//                     return false;
//                 }
//             } else {
//                 return false;
//             }
//         }
//         true
//     }
// }

// impl<T: Eq> Eq for Arr<T> {}

#[derive(Default)]
struct Bucket<T> {
    entries: SharePtr<MaybeUninit<T>>,
}

impl<T> Bucket<T> {
    #[inline(always)]
    const fn new(entries: *mut MaybeUninit<T>) -> Self {
        Bucket {
            entries: SharePtr::new(entries),
        }
    }
    fn alloc(len: usize) -> *mut MaybeUninit<T> {
        let entries = Vec::with_capacity(len);
        entries.into_raw_parts().0
    }
    fn init(&self, len: usize, lock: &ShareMutex<()>) -> *mut MaybeUninit<T> {
        let _lock = lock.lock();
        let mut ptr = self.entries.load(Ordering::Relaxed);
        if ptr.is_null() {
            ptr = Bucket::alloc(len);
            self.entries.store(ptr, Ordering::Relaxed);
        }
        ptr
    }
    fn to_vec(&self, len: usize) -> Vec<MaybeUninit<T>> {
        let ptr = self.entries.swap(ptr::null_mut(), Ordering::Relaxed);
        unsafe { Vec::from_raw_parts(ptr, len, len) }
    }
}

#[derive(Default)]
pub struct BlobArr(Arr<u8>);

impl BlobArr {
    pub fn new(blob_size: usize) -> BlobArr {
        BlobArr::with_capacity(0, blob_size)
    }

    pub fn with_capacity(capacity: usize, blob_size: usize) -> BlobArr {
        let mut buckets = [ptr::null_mut(); BUCKETS];
        if capacity == 0 {
            return BlobArr(Arr {
                buckets: buckets.map(Bucket::new),
                lock: ShareMutex::default(),
            });
        }
        let end = Location::of(capacity).bucket;
        for (i, bucket) in buckets[..=end].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = Bucket::alloc(len * blob_size);
        }

        BlobArr(Arr {
            buckets: buckets.map(Bucket::new),
            lock: ShareMutex::default(),
        })
    }
    #[inline(always)]
    pub fn get(&self, location: &Location, blob_size: usize) -> Option<&u8> {
        // safety: `location.bucket` is always in bounds
        let entries = unsafe { self.0.entries(location.bucket) };

        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }

        // safety: `location.entry` is always in bounds for it's bucket
        Some(unsafe { transmute(entries.add(location.entry * blob_size)) })
    }
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, location: &Location, blob_size: usize) -> &mut u8 {
        // safety: caller guarantees the entry is initialized
        transmute(
            self.0
                .entries(location.bucket)
                .add(location.entry * blob_size),
        )
    }
    #[inline(always)]
    pub fn get_mut(&mut self, location: &Location, blob_size: usize) -> Option<&mut u8> {
        let entries = unsafe { self.0.entries_mut(location.bucket) };
        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }
        // safety: `location.entry` is always in bounds for it's bucket
        Some(unsafe { transmute(entries.add(location.entry * blob_size)) })
    }
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, location: &Location, blob_size: usize) -> &mut u8 {
        transmute(
            self.0
                .entries_mut(location.bucket)
                .add(location.entry * blob_size),
        )
    }
    #[inline(always)]
    pub unsafe fn alloc(&mut self, location: &Location, blob_size: usize) -> &mut u8 {
        let bucket = self.0.buckets.get_unchecked_mut(location.bucket);
        // safety: `location.bucket` is always in bounds
        let mut entries = *bucket.entries.get_mut();
        // bucket is uninitialized
        if entries.is_null() {
            entries = Bucket::alloc(location.len * blob_size);
        }
        // safety: `location.entry` is always in bounds for it's bucket
        transmute(entries.add(location.entry * blob_size))
    }
    #[inline(always)]
    pub unsafe fn load_alloc(&self, location: Location, blob_size: usize) -> &mut u8 {
        let bucket = self.0.buckets.get_unchecked(location.bucket);
        // safety: `location.bucket` is always in bounds
        let mut entries = bucket.entries.load(Ordering::Relaxed);
        // bucket is uninitialized
        if entries.is_null() {
            entries = bucket.init(location.len * blob_size, &self.0.lock);
        }
        // safety: `location.entry` is always in bounds for it's bucket
        transmute(entries.add(location.entry * blob_size))
    }
    pub fn replace(&self, blob_size: usize) -> [Vec<MaybeUninit<u8>>; BUCKETS] {
        let mut buckets = [0; BUCKETS].map(|_| Vec::new());
        for (i, bucket) in self.0.buckets.iter().enumerate() {
            *unsafe { buckets.get_unchecked_mut(i) } =
                bucket.to_vec(Location::bucket_len(i) * blob_size);
        }
        buckets
    }
    pub fn iter(&self, blob_size: usize) -> BlobIter<'_> {
        self.slice(0..MAX_ENTRIES, blob_size)
    }
    pub fn slice(&self, range: Range<usize>, blob_size: usize) -> BlobIter<'_> {
        let mut it = Iter::new(&self.0.buckets, range);
        if !it.ptr.is_null() {
            it.ptr = unsafe { it.ptr.add(it.start.entry * blob_size) };
        } else {
            it.start.len = it.start.entry;
        }
        BlobIter { it, blob_size }
    }
}

pub struct BlobIter<'a> {
    it: Iter<'a, u8>,
    blob_size: usize,
}
impl<'a> BlobIter<'a> {
    #[inline(always)]
    pub fn empty() -> Self {
        BlobIter {
            it: Iter::empty(),
            blob_size: 0,
        }
    }
    #[inline(always)]
    pub fn blob_size(&self) -> usize {
        self.blob_size
    }
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.it.index()
    }
    #[inline(always)]
    pub(crate) fn step(&mut self) -> &'a mut u8 {
        unsafe {
            let ptr = self.it.ptr.add(self.blob_size);
            transmute(replace(&mut self.it.ptr, ptr))
        }
    }
}

impl<'a> Iterator for BlobIter<'a> {
    type Item = &'a mut u8;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.it.start.entry < self.it.start.len {
            self.it.start.entry += 1;
            return Some(self.step());
        }
        unsafe { transmute(self.it.next_bucket()) }
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

#[derive(Debug, Default)]
pub struct Location {
    // the index of the bucket
    bucket: usize,
    // the length
    len: usize,
    // the index of the entry in `bucket`
    entry: usize,
}

// skip the shorter buckets to avoid unnecessary allocations.
// this also reduces the maximum capacity of a arr.
const SKIP: usize = 32;
const SKIP_BUCKET: usize = ((usize::BITS - SKIP.leading_zeros()) as usize) - 1;

impl Location {
    #[inline(always)]
    pub const fn of(index: usize) -> Location {
        let skipped = index.checked_add(SKIP).expect("exceeded maximum length");
        let bucket = usize::BITS - skipped.leading_zeros();
        let bucket = (bucket as usize) - (SKIP_BUCKET + 1);
        let bucket_len = Location::bucket_len(bucket);
        let entry = skipped ^ bucket_len;

        Location {
            bucket,
            len: bucket_len,
            entry,
        }
    }
    #[inline(always)]
    const fn bucket_len(bucket: usize) -> usize {
        1 << (bucket + SKIP_BUCKET)
    }
    #[inline(always)]
    pub fn bucket(&self) -> usize {
        self.bucket
    }
    #[inline(always)]
    pub fn entry(&self) -> usize {
        self.entry
    }
    #[inline(always)]
    pub fn index(&self) -> usize {
        ((u32::MAX as u64) >> (u32::BITS - self.bucket as u32) << SKIP_BUCKET) as usize + self.entry
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use test::Bencher;

    use crate::*;
    static mut AAA: u64 = 0;
    #[test]
    fn test1() {
        let mut arr: Arr<u8> = Default::default();
        arr.alloc(&Location::of(0)).write(1);
        arr.alloc(&Location::of(1)).write(3);
        unsafe {
            assert_eq!(arr[0].as_ptr().read(), 1);
            assert_eq!(arr[1].as_ptr().read(), 3);
        }
    }
    #[test]
    fn test2() {
        unsafe {
            let mut arr: Arr<u8> = Default::default();
            arr.alloc(&Location::of(0)).write(1);
            arr.alloc(&Location::of(1)).write(2);
            arr.alloc(&Location::of(2)).write(4);
            arr.alloc(&Location::of(98)).write(98);
            let mut iterator = arr.iter();
            assert_eq!(iterator.size_hint().0, 32);
            let r = iterator.next().unwrap().as_ptr().read();
            assert_eq!((iterator.index() - 1, r), (0, 1));
            let r = iterator.next().unwrap().as_ptr().read();
            assert_eq!((iterator.index() - 1, r), (1, 2));
            let r = iterator.next().unwrap().as_ptr().read();
            assert_eq!((iterator.index() - 1, r), (2, 4));
            for i in 3..32 {
                let _ = iterator.next().unwrap().as_ptr().read();
                assert_eq!((iterator.index() - 1), (i));
            }
            for i in 96..98 {
                let _r = iterator.next().unwrap();
                assert_eq!((iterator.index() - 1), (i));
            }
            let r = iterator.next().unwrap().as_ptr().read();
            assert_eq!((iterator.index() - 1, r), (98, 98));
            for i in 99..224 {
                let _r = iterator.next().unwrap();
                assert_eq!((iterator.index() - 1), (i));
            }
            assert_eq!(iterator.next().is_none(), true);
            assert_eq!(iterator.size_hint().0, 0);
        }
    }
    #[test]
    fn test() {
        unsafe {
            let arr = arr![1; 3];
            assert_eq!(arr[0].assume_init_read(), 1);
            assert_eq!(arr[1].assume_init_read(), 1);
            assert_eq!(arr[2].assume_init_read(), 1);

            let mut arr = arr![0];
            arr.alloc(&Location::of(1)).write(1);
            arr.alloc(&Location::of(2)).write(2);

            let mut arr = arr![10, 40, 30];
            assert_eq!(40, *arr.alloc(&Location::of(1)).as_ptr());

            {
                let arr: Arr<i8> = arr![10, 40, 30];
                assert_eq!(40, arr.get(&Location::of(1)).unwrap().assume_init_read());
                assert_eq!(true, arr.get(&Location::of(33)).is_none());
            }

            let arr = crate::arr![1, 2, 4];
            assert_eq!(*arr.get_unchecked(&Location::of(1)).as_ptr(), 2);

            let mut arr = crate::arr![1, 2, 4];
            assert_eq!(*arr.get_unchecked_mut(&Location::of(1)).as_ptr(), 2);

            let arr = arr![10, 40, 30];
            assert_eq!(40, *arr.load_alloc(&Location::of(1)).as_ptr());

            let arr = arr![10, 40, 30];
            assert_eq!(40, *arr.load(&Location::of(1)).unwrap().as_ptr());

            let arr = crate::arr![1, 2];
            arr.load_alloc(&Location::of(2)).write(3);
            assert_eq!(*arr[0].as_ptr(), 1);
            assert_eq!(*arr[1].as_ptr(), 2);
            assert_eq!(*arr[2].as_ptr(), 3);

            let arr = crate::arr![1, 2, 4];
            arr.load_alloc(&Location::of(98)).write(98);

            let mut iterator = arr.slice(0..160).enumerate();
            assert_eq!(iterator.size_hint().0, 32);
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1.as_ptr()), (0, 1));
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1.as_ptr()), (1, 2));
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1.as_ptr()), (2, 4));
            for i in 3..32 {
                let r = iterator.next().unwrap();
                assert_eq!((r.0), (i));
            }
            for i in 32..34 {
                let r = iterator.next().unwrap();
                assert_eq!((r.0), (i));
            }
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1.as_ptr()), (34, 98));
            for i in 35..96 {
                let r = iterator.next().unwrap();
                assert_eq!((r.0), (i));
            }
            assert_eq!(iterator.next().is_none(), true);
            assert_eq!(iterator.size_hint().0, 0);

            let mut iterator = arr.slice(1..3).enumerate();
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1.as_ptr()), (0, 2));
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1.as_ptr()), (1, 4));
            assert_eq!(iterator.next().is_none(), true);
        }
    }
    #[test]
    fn test_arc() {
        unsafe {
            let arr = Arc::new(crate::Arr::new());

            // spawn 6 threads that append to the arr
            let threads = (0..6)
                .map(|i| {
                    let arr = arr.clone();

                    std::thread::spawn(move || {
                        arr.load_alloc(&Location::of(i)).write(i);
                    })
                })
                .collect::<Vec<_>>();

            // wait for the threads to finish
            for thread in threads {
                thread.join().unwrap();
            }

            for i in 0..6 {
                assert!(arr.iter().any(|x| *x.as_ptr() == i));
            }
        }
    }
    #[test]
    fn test_arc1() {
        unsafe {
            let mut arr = Arr::with_capacity(0);
            let _a1 = {
                for i in 0..6 {
                    arr.alloc(&Location::of(i)).write(Some(Arc::new(i)));
                }
                for i in 0..6 {
                    assert_eq!(arr[i].assume_init_ref().as_ref().unwrap().as_ref(), &i);
                }
                arr[1].assume_init_ref().as_ref().unwrap().clone()
            };
            assert_eq!(
                Arc::<usize>::strong_count(arr[0].assume_init_ref().as_ref().unwrap()),
                1
            );
            assert_eq!(
                Arc::<usize>::strong_count(arr[1].assume_init_ref().as_ref().unwrap()),
                2
            );
        }
    }
    #[test]
    fn test_mutex() {
        unsafe {
            let arr: Arc<Arr<Option<Mutex<u32>>>> = Arc::new(crate::Arr::default());

            // set an element
            let r = arr.load_alloc(&Location::of(0));
            r.write(Some(Mutex::new(1)));

            let thread = std::thread::spawn({
                let arr = arr.clone();
                move || {
                    // mutate through the mutex
                    *(arr[0].assume_init_ref().as_ref().unwrap().lock().unwrap()) += 1;
                }
            });

            thread.join().unwrap();
            let r = arr[0].assume_init_ref().as_ref();
            let x = r.unwrap().lock().unwrap();
            assert_eq!(*x, 2);
        }
    }
    #[test]
    fn location() {
        assert_eq!(Location::bucket_len(0), 32);
        for i in 0..32 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 32);
            assert_eq!(loc.bucket, 0);
            assert_eq!(loc.entry, i);
            assert_eq!(loc.index(), i)
        }

        assert_eq!(Location::bucket_len(1), 64);
        for i in 33..96 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 64);
            assert_eq!(loc.bucket, 1);
            assert_eq!(loc.entry, i - 32);
            assert_eq!(loc.index(), i)
        }

        assert_eq!(Location::bucket_len(2), 128);
        for i in 96..224 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 128);
            assert_eq!(loc.bucket, 2);
            assert_eq!(loc.entry, i - 96);
            assert_eq!(loc.index(), i)
        }

        let max = Location::of(MAX_ENTRIES);
        assert_eq!(max.bucket, BUCKETS - 1);
        assert_eq!(max.len, 1 << 31);
        assert_eq!(max.entry, (1 << 31) - 1);
    }
    #[bench]
    fn bench_loc(b: &mut Bencher) {
        b.iter(move || {
            for i in 0..1000 {
                unsafe { AAA += Location::of(i).entry() as u64 };
            }
        });
    }
    #[bench]
    fn bench_arr(b: &mut Bencher) {
        let mut arr = Arr::with_capacity(0);
        for i in 0..2000 {
            arr.alloc(&Location::of(i)).write(0u64);
        }
        b.iter(move || {
            for i in arr.slice(100..200) {
                unsafe { AAA += i.assume_init_read() };
            }
        });
    }
    #[bench]
    fn bench_vec(b: &mut Bencher) {
        let mut arr: Vec<u64> = Vec::with_capacity(0);
        for _ in 0..100 {
            arr.push(0u64);
        }
        b.iter(move || {
            for i in arr.iter() {
                unsafe { AAA += *i };
            }
        });
    }
    #[bench]
    fn bench_vec1(b: &mut Bencher) {
        let mut arrs = [0; 1].map(|_| Vec::with_capacity(0));
        for _ in 0..1000 {
            for j in arrs.iter_mut() {
                j.push(0u64);
            }
        }
        let mut c = 0u64;
        b.iter(move || {
            for i in 0..800 {
                for j in arrs.iter() {
                    c += unsafe { j.get_unchecked(i) };
                }
            }
        });
        assert_eq!(c, 0);
    }
}
