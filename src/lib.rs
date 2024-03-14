//! 数组，使用多个槽，每个槽用不扩容的Vec来装元素，当槽位上的Vec长度不够时，不会扩容Vec，而是线程安全的到下一个槽位分配新Vec。
//! 第一个槽位的Vec长度为32。
//! 迭代性能比Vec慢1-10倍， 主要损失在切换bucket时，原子操作及缓存失效。

#![feature(vec_into_raw_parts)]
#![feature(const_option)]
#![feature(test)]
extern crate test;

use std::mem::{forget, replace, transmute};
use std::ops::{Index, IndexMut, Range};
use std::ptr;
use std::sync::atomic::Ordering;

use pi_null::Null;
use pi_share::{ShareMutex, SharePtr};

pub const BUCKETS: usize = (u32::BITS as usize) - SKIP_BUCKET;
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
/// assert_eq!(arr[0], 1);
/// assert_eq!(arr[1], 2);
/// assert_eq!(arr[2], 3);
/// ```
///
/// - Create a [`Arr`] from a given element and size:
///
/// ```
/// let arr = pi_arr::arr![1; 3];
/// assert_eq!(arr[0], 1);
/// assert_eq!(arr[1], 1);
/// assert_eq!(arr[2], 1);
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
    buckets: [SharePtr<T>; BUCKETS],
    lock: ShareMutex<()>,
}

unsafe impl<T: Send> Send for Arr<T> {}
unsafe impl<T: Sync> Sync for Arr<T> {}

impl<T: Null> Arr<T> {
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
    /// let mut arr = pi_arr::Arr::with_capacity(10);
    ///
    /// for i in 0..32 {
    ///     // will not allocate
    ///     arr.set(i, i);
    /// }
    ///
    /// // may allocate
    /// arr.set(33, 33);
    /// ```
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Arr<T> {
        Self::with_capacity_multiple(capacity, 1)
    }
    #[inline(always)]
    pub fn with_capacity_multiple(capacity: usize, multiple: usize) -> Arr<T> {
        let mut buckets = [ptr::null_mut(); BUCKETS];
        if capacity == 0 {
            return Arr {
                buckets: buckets.map(SharePtr::new),
                lock: ShareMutex::default(),
            };
        }
        let end = Location::of(capacity).bucket as usize;
        for (i, bucket) in buckets[..=end].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = bucket_alloc(len * multiple);
        }

        Arr {
            buckets: buckets.map(SharePtr::new),
            lock: ShareMutex::default(),
        }
    }

    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&40), arr.get(1));
    /// assert_eq!(None, arr.get(33));
    /// ```
    #[inline(always)]
    pub fn get(&self, location: &Location) -> Option<&T> {
        // safety: `location.bucket` is always in bounds
        let entries = unsafe { self.entries(location.bucket as usize) };

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
    ///     assert_eq!(arr.get_unchecked(1), &2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, location: &Location) -> &T {
        &*self.entries(location.bucket as usize).add(location.entry)
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&mut 40), arr.get_mut(1));
    /// assert_eq!(None, arr.get_mut(33));
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, location: &Location) -> Option<&mut T> {
        let entries = unsafe { self.entries(location.bucket as usize) };

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
    ///     assert_eq!(arr.get_unchecked_mut(1), &mut 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, location: &Location) -> &mut T {
        &mut *self.entries(location.bucket as usize).add(location.entry)
    }
    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is null
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.alloc(1, 1));
    /// assert_eq!(true, arr.alloc(3, 1).is_null());
    /// ```
    #[inline(always)]
    pub fn alloc(&mut self, location: &Location) -> &mut T {
        let entries = self.alloc_bucket(location);
        // safety: `location.entry` is always in bounds for it's bucket
        unsafe { &mut *entries.add(location.entry) }
    }
    /// set element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, arr.set(&Location::of(1), 20));
    /// assert_eq!(Some(&20), arr.get(&Location::of(1)));
    /// assert_eq!(true, arr.set(&Location::of(33), 5).is_null());
    /// assert_eq!(Some(&5), arr.get(&Location::of(33)));
    /// ```
    #[inline(always)]
    pub fn set(&mut self, location: &Location, value: T) -> T {
        replace(self.alloc(location), value)
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will not be allocated automatically, and the returned None.
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(10, *arr.load(&Location::of(0)).unwrap());
    /// assert_eq!(Some(&mut 40), arr.load(1));
    /// assert_eq!(true, arr.load(3).unwrap().is_null());
    /// assert_eq!(None, arr.load(33));
    /// ```
    #[inline(always)]
    pub fn load(&self, location: &Location) -> Option<&mut T> {
        let entries = unsafe { self.load_entries(location.bucket as usize) };

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
    ///     assert_eq!(arr.load_unchecked(1), &mut 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn load_unchecked(&self, location: &Location) -> &mut T {
        &mut *self
            .load_entries(location.bucket as usize)
            .add(location.entry)
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is null
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.load_alloc(1, 1));
    /// assert_eq!(true, arr.load_alloc(3,1).is_null());
    /// ```
    #[inline(always)]
    pub fn load_alloc(&self, location: &Location) -> &mut T {
        let entries = self.load_alloc_bucket(location);
        // safety: `location.entry` is always in bounds for it's bucket
        unsafe { transmute(entries.add(location.entry)) }
    }
    /// insert an element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2];
    /// arr.insert(2, 3);
    /// assert_eq!(arr[0], 1);
    /// assert_eq!(arr[1], 2);
    /// assert_eq!(arr[2], 3);
    /// ```
    #[inline(always)]
    pub fn insert(&self, location: &Location, value: T) -> T {
        replace(self.load_alloc(location), value)
    }

    /// replace buckets.
    pub fn replace(&self) -> [*mut T; BUCKETS] {
        let buckets = [0; BUCKETS].map(|_| ptr::null_mut());
        for bucket in self.buckets.iter() {
            bucket.swap(ptr::null_mut(), Ordering::Relaxed);
        }
        buckets
    }

    /// Returns an iterator over the array.
    ///
    /// Values are yielded in the form `Entry`. The array may
    /// have in-progress concurrent writes that create gaps, so `index`
    /// may not be strictly sequential.
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let arr = pi_arr::arr![1, 2, 4];
    /// arr.insert(98, 98);
    /// let mut iterator = arr.iter();
    /// assert_eq!(iterator.size_hint().0, 32);
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (0, 1));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (1, 2));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (2, 4));
    /// for i in 3..32 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1, *r), (i, i32::null()));
    /// }
    /// for i in 96..98 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1, *r), (i, i32::null()));
    /// }
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (98, 98));
    /// for i in 99..224 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1, *r), (i, i32::null()));
    /// }
    /// assert_eq!(iterator.next(), None);
    /// assert_eq!(iterator.size_hint().0, 0);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.slice(0..MAX_ENTRIES)
    }

    /// Returns an iterator over the array at the given range.
    ///
    /// Values are yielded in the form `Entry`.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2, 4, 6];
    /// let mut iterator = arr.slice(1..3);
    ///
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 2);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 4);
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline(always)]
    pub fn slice(&self, range: Range<usize>) -> Iter<'_, T> {
        Iter::new(&self.buckets, range).init_iter()
    }
    /// Returns an reverse iterator over the array at the given range.
    ///
    /// Values are yielded in the form `Entry`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let arr = pi_arr::arr![1, 2, 4, 6];
    /// let mut iterator = arr.reverse_iter();
    ///
    /// for i in 4..32 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!(32 - i + 3, iterator.index());
    ///     assert_eq!(*r, u32::null());
    /// }
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 6);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 4);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 2);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 1);
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline]
    pub fn reverse_iter(&self) -> ReverseIter<'_, T> {
        self.reverse_slice(0..MAX_ENTRIES)
    }
    /// Returns an iterator over the array at the given range.
    ///
    /// Values are yielded in the form `Entry`.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2, 4, 6];
    /// let mut iterator = arr.reverse_slice(1..3);
    ///
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 4);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 2);
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline(always)]
    pub fn reverse_slice(&self, range: Range<usize>) -> ReverseIter<'_, T> {
        Iter::new(&self.buckets, range).init_reverse()
    }
    #[inline(always)]
    pub fn iter_with_ptr(&self, ptr: *mut T, start: Location, end: Location) -> Iter<'_, T> {
        Iter {
            buckets: &self.buckets,
            start,
            end,
            ptr,
        }
    }

    #[inline(always)]
    pub unsafe fn load_entries(&self, bucket: usize) -> *mut T {
        self.buckets.get_unchecked(bucket).load(Ordering::Relaxed)
    }
    #[inline(always)]
    pub unsafe fn entries(&self, bucket: usize) -> *mut T {
        *self.buckets.get_unchecked(bucket).as_ptr()
    }
    #[inline(always)]
    pub unsafe fn entries_mut(&mut self, bucket: usize) -> *mut T {
        *self.buckets.get_unchecked_mut(bucket).get_mut()
    }
    #[inline(always)]
    pub fn load_alloc_bucket(&self, location: &Location) -> *mut T {
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket as usize) };
        // safety: `location.bucket` is always in bounds
        let mut entries = bucket.load(Ordering::Relaxed);
        // bucket is uninitialized
        if entries.is_null() {
            entries = bucket_init(bucket, location.len, &self.lock)
        }
        entries
    }
    #[inline(always)]
    pub fn alloc_bucket(&mut self, location: &Location) -> *mut T {
        let bucket = unsafe { self.buckets.get_unchecked_mut(location.bucket as usize) };
        // safety: `location.bucket` is always in bounds
        let mut entries = *bucket.get_mut();

        // bucket is uninitialized
        if entries.is_null() {
            entries = bucket_init(bucket, location.len, &self.lock);
        }
        entries
    }
    #[inline(always)]
    pub fn buckets(&self) -> &[SharePtr<T>] {
        &self.buckets
    }
}

impl<T: Null> Index<usize> for Arr<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(&Location::of(index))
            .expect("no element found at index {index}")
    }
}
impl<T: Null> IndexMut<usize> for Arr<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(&Location::of(index))
            .expect("no element found at index_mut {index}")
    }
}
impl<T> Drop for Arr<T> {
    fn drop(&mut self) {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let entries = *bucket.get_mut();
            if entries.is_null() {
                continue;
            }
            let len = Location::bucket_len(i);
            // safety: in drop
            unsafe { drop(Vec::from_raw_parts(entries, len, len)) }
        }
    }
}

impl<T: Null> FromIterator<T> for Arr<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let mut arr = Arr::with_capacity(lower);
        for (i, value) in iter.enumerate() {
            arr.set(&Location::of(i), value);
        }
        arr
    }
}

impl<T: Null> Extend<T> for Arr<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        for (i, value) in iter.enumerate() {
            self.set(&Location::of(i), value);
        }
    }
}

impl<T: Null + Clone> Clone for Arr<T> {
    fn clone(&self) -> Arr<T> {
        let mut buckets: [*mut T; BUCKETS] = [ptr::null_mut(); BUCKETS];

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
            buckets: buckets.map(SharePtr::new),
            lock: ShareMutex::default(),
        }
    }
}

/// An iterator over the elements of a [`Arr<T>`].
///
/// See [`Arr::iter`] for details.

pub struct Iter<'a, T> {
    buckets: &'a [SharePtr<T>],
    start: Location,
    end: Location,
    ptr: *mut T,
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
    #[inline(always)]
    fn new(buckets: &'a [SharePtr<T>], range: Range<usize>) -> Self {
        let start = Location::of(range.start);
        let end = Location::of(range.end);
        Iter {
            buckets,
            start,
            end,
            ptr: ptr::null_mut(),
        }
    }
    #[inline(always)]
    fn init_iter(mut self) -> Self {
        let ptr = unsafe {
            self.buckets
                .get_unchecked(self.start.bucket as usize)
                .load(Ordering::Relaxed)
        };
        if ptr.is_null() || self.start.bucket > self.end.bucket {
            self.start.len = self.start.entry;
        } else if self.start.bucket == self.end.bucket {
            self.start.len = self.end.entry;
        }
        self.ptr = ptr;
        self
    }
    #[inline(always)]
    fn init_reverse(mut self) -> ReverseIter<'a, T> {
        let ptr = unsafe {
            self.buckets
                .get_unchecked(self.end.bucket as usize)
                .load(Ordering::Relaxed)
        };
        if ptr.is_null() || self.start.bucket > self.end.bucket {
            self.end.len = self.end.entry;
        } else if self.start.bucket == self.end.bucket {
            self.end.len = self.start.entry;
        } else {
            self.end.len = 0;
        }
        self.ptr = ptr;
        ReverseIter(self)
    }
    #[inline(always)]
    pub fn start(&self) -> &Location {
        &self.start
    }
    #[inline(always)]
    pub fn end(&self) -> &Location {
        &self.end
    }
    #[inline(always)]
    pub(crate) fn get(&mut self) -> &'a mut T {
        unsafe { transmute(self.ptr.add(self.start.entry)) }
    }
    #[inline]
    pub(crate) fn next_bucket(&mut self) -> Option<&'a mut T> {
        loop {
            if self.start.bucket >= self.end.bucket {
                return None;
            }
            self.start.bucket += 1;
            self.ptr = unsafe {
                transmute(
                    self.buckets
                        .get_unchecked(self.start.bucket as usize)
                        .load(Ordering::Relaxed),
                )
            };
            if !self.ptr.is_null() {
                if self.start.bucket == self.end.bucket {
                    self.start.len = self.end.entry;
                } else {
                    self.start.len = Location::bucket_len(self.start.bucket as usize);
                }
                self.start.entry = 1;
                return Some(unsafe { transmute(self.ptr) });
            }
        }
    }
    #[inline(always)]
    pub(crate) fn get_last(&mut self) -> &'a mut T {
        unsafe { transmute(self.ptr.add(self.end.entry)) }
    }
    #[inline]
    pub(crate) fn prev_bucket(&mut self) -> Option<&'a mut T> {
        loop {
            if self.start.bucket >= self.end.bucket {
                return None;
            }
            self.end.bucket -= 1;
            self.ptr = unsafe {
                transmute(
                    self.buckets
                        .get_unchecked(self.end.bucket as usize)
                        .load(Ordering::Relaxed),
                )
            };
            if !self.ptr.is_null() {
                self.end.entry = Location::bucket_len(self.end.bucket as usize) - 1;
                if self.start.bucket == self.end.bucket {
                    self.end.len = self.start.entry;
                } else {
                    self.end.len = 0;
                }
                return Some(self.get_last());
            }
        }
    }
    fn size(&self) -> (usize, Option<usize>) {
        if self.start.bucket > self.end.bucket {
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
    type Item = &'a mut T;
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
        if self.start.bucket == self.end.bucket {
            let n = self.start.len.saturating_sub(self.start.entry);
            return (n, Some(n));
        }
        self.size()
    }
}

pub struct ReverseIter<'a, T>(Iter<'a, T>);
impl<'a, T> ReverseIter<'a, T> {
    #[inline(always)]
    pub fn end(&self) -> &Location {
        self.0.end()
    }
}
impl<'a, T> Iterator for ReverseIter<'a, T> {
    type Item = &'a mut T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.end.entry > self.0.end.len {
            self.0.end.entry -= 1;
            let r = self.0.get_last();
            return Some(r);
        }
        self.0.prev_bucket()
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.0.start.bucket == self.0.end.bucket {
            let n = self.0.end.entry.saturating_sub(self.0.end.len);
            return (n, Some(n));
        }
        self.0.size()
    }
}

fn bucket_alloc<T: Null>(len: usize) -> *mut T {
    let mut entries: Vec<T> = Vec::with_capacity(len);
    entries.resize_with(entries.capacity(), || T::null());
    entries.into_raw_parts().0
}
fn bucket_init<T: Null>(share_ptr: &SharePtr<T>, len: usize, lock: &ShareMutex<()>) -> *mut T {
    let _lock = lock.lock();
    let mut ptr = share_ptr.load(Ordering::Relaxed);
    if ptr.is_null() {
        ptr = bucket_alloc(len);
        share_ptr.store(ptr, Ordering::Relaxed);
    }
    ptr
}

// skip the shorter buckets to avoid unnecessary allocations.
// this also reduces the maximum capacity of a arr.
const SKIP: usize = 32;
const SKIP_BUCKET: usize = ((usize::BITS - SKIP.leading_zeros()) as usize) - 1;

#[derive(Debug, Default, Clone)]
pub struct Location {
    // the index of the bucket
    pub bucket: isize,
    // the length
    pub len: usize,
    // the index of the entry in `bucket`
    pub entry: usize,
}

impl Location {
    #[inline(always)]
    pub const fn new(bucket: isize, bucket_len: usize, entry: usize) -> Self {
        Location {
            bucket,
            len: bucket_len,
            entry,
        }
    }
    #[inline(always)]
    pub const fn of(index: usize) -> Location {
        let skipped = index.checked_add(SKIP).expect("exceeded maximum length");
        let bucket = usize::BITS - skipped.leading_zeros();
        let bucket = (bucket as usize) - (SKIP_BUCKET + 1);
        let bucket_len = Location::bucket_len(bucket);
        let entry = skipped ^ bucket_len;

        Location {
            bucket: bucket as isize,
            len: bucket_len,
            entry,
        }
    }
    #[inline(always)]
    pub const fn bucket_len(bucket: usize) -> usize {
        1 << (bucket + SKIP_BUCKET)
    }
    #[inline(always)]
    pub const fn index(bucket: u32, entry: usize) -> usize {
        ((i32::MAX as u32) >> (u32::BITS - 1 - bucket) << SKIP_BUCKET) as usize + entry
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
        let arr: Arr<u8> = arr![1; 3];
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 1);
    }
    #[test]
    fn test2() {
        let arr = arr![1, 2, 4];
        arr.insert(&Location::of(98), 98);
        let mut iterator = arr.iter();
        assert_eq!(iterator.size().0, 32);
        let r = iterator.next().unwrap();
        assert_eq!(
            (
                Location::index(iterator.start().bucket as u32, iterator.start().entry - 1),
                *r
            ),
            (0, 1)
        );
        let r = iterator.next().unwrap();
        assert_eq!(
            (
                Location::index(iterator.start().bucket as u32, iterator.start().entry - 1),
                *r
            ),
            (1, 2)
        );
        let r = iterator.next().unwrap();
        assert_eq!(
            (
                Location::index(iterator.start().bucket as u32, iterator.start().entry - 1),
                *r
            ),
            (2, 4)
        );
        for i in 3..32 {
            let r = iterator.next().unwrap();
            assert_eq!(
                (
                    Location::index(iterator.start().bucket as u32, iterator.start().entry - 1),
                    *r
                ),
                (i, i32::null())
            );
        }
        for i in 96..98 {
            let r = iterator.next().unwrap();
            assert_eq!(
                (
                    Location::index(iterator.start().bucket as u32, iterator.start().entry - 1),
                    *r
                ),
                (i, i32::null())
            );
        }
        let r = iterator.next().unwrap();
        assert_eq!(
            (
                Location::index(iterator.start().bucket as u32, iterator.start().entry - 1),
                *r
            ),
            (98, 98)
        );
        for i in 99..224 {
            let r = iterator.next().unwrap();
            assert_eq!(
                (
                    Location::index(iterator.start().bucket as u32, iterator.start().entry - 1),
                    *r
                ),
                (i, i32::null())
            );
        }
        assert_eq!(iterator.next(), None);
        assert_eq!(iterator.size().0, 0);
    }
    #[test]
    fn test3() {
        let arr = arr![1, 2, 4, 6];
        let mut iterator = arr.reverse_iter();
        for i in 4..32 {
            let r = iterator.next().unwrap();
            assert_eq!(
                32 - i + 3,
                Location::index(iterator.end().bucket as u32, iterator.end().entry)
            );
            assert_eq!(*r, i32::null());
        }
        let r = iterator.next().unwrap();
        assert_eq!(*r, 6);
        let r = iterator.next().unwrap();
        assert_eq!(*r, 4);
        let r = iterator.next().unwrap();
        assert_eq!(*r, 2);
        let r = iterator.next().unwrap();
        assert_eq!(*r, 1);
        assert_eq!(iterator.next(), None);
    }
    #[test]
    fn test() {
        let arr = arr![1; 3];
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 1);

        let mut arr = arr![10, 40, 30];
        assert_eq!(40, *arr.alloc(&Location::of(1)));
        assert_eq!(true, arr.alloc(&Location::of(3)).is_null());
        assert_eq!(40, arr.set(&Location::of(1), 20));
        assert_eq!(true, arr.set(&Location::of(33), 33).is_null());

        {
            let arr: Arr<i8> = arr![10, 40, 30];
            assert_eq!(Some(&40), arr.get(&Location::of(1)));
            assert_eq!(None, arr.get(&Location::of(33)));
        }

        let arr = crate::arr![1, 2, 4];
        unsafe {
            assert_eq!(arr.get_unchecked(&Location::of(1)), &2);
        }

        let mut arr = arr![10, 40, 30];
        assert_eq!(Some(&mut 40), arr.get_mut(&Location::of(1)));
        assert_eq!(None, arr.get_mut(&Location::of(33)));

        let mut arr = crate::arr![1, 2, 4];
        unsafe {
            assert_eq!(arr.get_unchecked_mut(&Location::of(1)), &mut 2);
        }

        let arr = arr![10, 40, 30];
        assert_eq!(40, *arr.load_alloc(&Location::of(1)));
        assert_eq!(true, arr.load_alloc(&Location::of(3)).is_null());
        assert_eq!(true, arr.load_alloc(&Location::of(133)).is_null());

        let arr = arr![10, 40, 30];
        assert_eq!(40, *arr.load(&Location::of(1)).unwrap());
        assert_eq!(true, arr.load(&Location::of(3)).unwrap().is_null());
        assert_eq!(None, arr.load(&Location::of(133)));

        let arr = crate::arr![1, 2];
        arr.insert(&Location::of(2), 3);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);

        let arr = crate::arr![1, 2, 4];
        arr.insert(&Location::of(98), 98);
        let mut iterator = arr.slice(0..160).enumerate();
        assert_eq!(iterator.size_hint().0, 32);
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (0, 1));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (1, 2));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (2, 4));
        for i in 3..32 {
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1), (i, i32::null()));
        }
        for i in 32..34 {
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1), (i, i32::null()));
        }
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (34, 98));
        for i in 35..96 {
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1), (i, i32::null()));
        }
        assert_eq!(iterator.next(), None);
        assert_eq!(iterator.size_hint().0, 0);

        let mut iterator = arr.slice(1..3).enumerate();
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (0, 2));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (1, 4));
        assert_eq!(iterator.next(), None);
    }
    #[test]
    fn test_arc() {
        let arr = Arc::new(crate::Arr::new());

        // spawn 6 threads that append to the arr
        let threads = (0..6)
            .map(|i| {
                let arr = arr.clone();

                std::thread::spawn(move || {
                    arr.insert(&Location::of(i), i);
                })
            })
            .collect::<Vec<_>>();

        // wait for the threads to finish
        for thread in threads {
            thread.join().unwrap();
        }

        for i in 0..6 {
            assert!(arr.iter().any(|x| *x == i));
        }
    }
    #[test]
    fn test_arc1() {
        let a1 = {
            let arr = crate::Arr::new();
            for i in 0..6 {
                arr.insert(&Location::of(i), Some(Arc::new(i)));
            }

            for i in 0..6 {
                assert_eq!(arr[i].as_ref().unwrap().as_ref(), &i);
            }
            let a2 = arr.clone();
            assert_eq!(Arc::<usize>::strong_count(a2[0].as_ref().unwrap()), 2);
            a2
        };
        assert_eq!(Arc::<usize>::strong_count(a1[0].as_ref().unwrap()), 1);
    }
    #[test]
    fn test_mutex() {
        let arr = Arc::new(crate::Arr::new());

        // set an element
        arr.insert(&Location::of(0), Some(Mutex::new(1)));

        let thread = std::thread::spawn({
            let arr = arr.clone();
            move || {
                // mutate through the mutex
                *(arr[0].as_ref().unwrap().lock().unwrap()) += 1;
            }
        });

        thread.join().unwrap();

        let x = arr[0].as_ref().unwrap().lock().unwrap();
        assert_eq!(*x, 2);
    }
    #[test]
    fn location() {
        assert_eq!(Location::bucket_len(0), 32);
        for i in 0..32 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 32);
            assert_eq!(loc.bucket, 0);
            assert_eq!(loc.entry, i);
            assert_eq!(Location::index(loc.bucket as u32, loc.entry), i)
        }

        assert_eq!(Location::bucket_len(1), 64);
        for i in 33..96 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 64);
            assert_eq!(loc.bucket, 1);
            assert_eq!(loc.entry, i - 32);
            assert_eq!(Location::index(loc.bucket as u32, loc.entry), i)
        }

        assert_eq!(Location::bucket_len(2), 128);
        for i in 96..224 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 128);
            assert_eq!(loc.bucket, 2);
            assert_eq!(loc.entry, i - 96);
            assert_eq!(Location::index(loc.bucket as u32, loc.entry), i)
        }

        let max = Location::of(MAX_ENTRIES);
        assert_eq!(max.bucket as usize, BUCKETS - 1);
        assert_eq!(max.len, 1 << 31);
        assert_eq!(max.entry, (1 << 31) - 1);
    }
    #[bench]
    fn bench_loc(b: &mut Bencher) {
        b.iter(move || {
            for i in 0..1000 {
                unsafe { AAA += Location::of(i).entry as u64 };
            }
        });
    }
    #[bench]
    fn bench_arr(b: &mut Bencher) {
        let mut arr = Arr::new();
        for i in 0..2000 {
            *arr.alloc(&Location::of(i)) = 0;
        }
        b.iter(move || {
            for i in arr.slice(100..200) {
                unsafe { AAA += *i };
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
            for i in 0..100 {
                for j in arrs.iter() {
                    c += unsafe { j.get_unchecked(i) };
                }
            }
        });
        assert_eq!(c, 0);
    }
}
