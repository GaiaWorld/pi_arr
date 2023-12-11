#![feature(vec_into_raw_parts)]
#![feature(const_option)]

use std::marker::PhantomData;
use std::mem::{forget, replace, size_of, transmute};
use std::ops::{Index, IndexMut, Range};
use std::sync::atomic::Ordering;
use std::{fmt, ptr};

use pi_null::Null;
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
pub struct Arr<T: Null> {
    raw: RawArr,
    _k: PhantomData<T>,
}

unsafe impl<T: Send + Null> Send for Arr<T> {}
unsafe impl<T: Sync + Null> Sync for Arr<T> {}

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
        Arr {
            raw: RawArr::with_capacity(capacity, Self::initialize, size_of::<T>()),
            _k: PhantomData,
        }
    }
    /// Returns the number of elements in the array.
    ///
    /// Since there is a default value, a bucket is allocated,
    /// and the length is automatically increased by the length of the bucket.
    /// Bucket lengths start at 32 as powers of 2.
    /// # Examples
    ///
    /// ```
    /// let mut arr = pi_arr::Arr::new();
    /// assert_eq!(arr.len(), 0);
    /// arr.set(1, 1);
    /// arr.set(2, 2);
    /// assert_eq!(arr.len(), 32);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.raw.len()
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
    pub fn get(&self, index: usize) -> Option<&T> {
        unsafe { transmute(self.raw.get(index, size_of::<T>())) }
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
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        transmute(self.raw.get_unchecked(index, size_of::<T>()))
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
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        unsafe { transmute(self.raw.get_mut(index, size_of::<T>())) }
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
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        transmute(self.raw.get_unchecked_mut(index, size_of::<T>()))
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
    /// assert_eq!(40, *arr.get_alloc(1));
    /// assert_eq!(true, arr.get_alloc(3).is_null());
    /// ```
    #[inline(always)]
    pub fn get_alloc(&mut self, index: usize) -> &mut T {
        unsafe { transmute(self.raw.get_alloc(index, Self::initialize, size_of::<T>())) }
    }
    /// set element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, arr.set(1, 20));
    /// assert_eq!(Some(&20), arr.get(1));
    /// assert_eq!(true, arr.set(33, 5).is_null());
    /// assert_eq!(Some(&5), arr.get(33));
    /// ```
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: T) -> T {
        replace(self.get_alloc(index), value)
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
    /// assert_eq!(10, *arr.load(0).unwrap());
    /// assert_eq!(Some(&mut 40), arr.load(1));
    /// assert_eq!(true, arr.load(3).unwrap().is_null());
    /// assert_eq!(None, arr.load(33));
    /// ```
    #[inline(always)]
    pub fn load(&self, index: usize) -> Option<&mut T> {
        unsafe { transmute(self.raw.get(index, size_of::<T>())) }
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
    pub unsafe fn load_unchecked(&self, index: usize) -> &mut T {
        transmute(self.raw.get_unchecked(index, size_of::<T>()))
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is null
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.load_alloc(1));
    /// assert_eq!(true, arr.load_alloc(3).is_null());
    /// ```
    #[inline(always)]
    pub fn load_alloc(&self, index: usize) -> &mut T {
        unsafe { transmute(self.raw.load_alloc(index, Self::initialize, size_of::<T>())) }
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
    pub fn insert(&self, index: usize, value: T) -> T {
        replace(self.load_alloc(index), value)
    }
    /// clear all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let mut arr = pi_arr::arr![1, 2];
    /// unsafe {arr.clear()};
    /// arr.set(2, 3);
    /// assert_eq!(arr[0], i32::null());
    /// assert_eq!(arr[1], i32::null());
    /// assert_eq!(arr[2], 3);
    /// ```
    pub unsafe fn clear(&self) {
        for (i, bucket) in self.raw.buckets.iter().enumerate() {
            let entries = bucket.entries.load(Ordering::Acquire);
            if entries.is_null() {
                continue;
            }
            let len = Location::bucket_len(i);
            // safety: in clear
            let mut vec = unsafe { Vec::from_raw_parts(entries as *mut T, len, len) };
            vec.clear();
            forget(vec);
            Self::initialize(entries, size_of::<T>(), len);
        }
    }
    /// clear and free buckets.
    ///
    /// # Examples
    ///
    /// ```
    /// use pi_null::Null;
    /// let mut arr = pi_arr::arr![1, 2];
    /// unsafe {arr.free()};
    /// arr.set(2, 3);
    /// assert_eq!(arr[0], i32::null());
    /// assert_eq!(arr[1], i32::null());
    /// assert_eq!(arr[2], 3);
    /// ```
    pub unsafe fn free(&self) {
        for (entries, len) in self.raw.replace().into_iter() {
            if entries.is_null() {
                continue;
            }
            // safety: in drop
            unsafe { drop(Vec::from_raw_parts(entries as *mut T, len, len)) }
        }
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
    /// Values are yielded in the form `(index, Entry)`.
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
        Iter {
            raw: self.raw.slice(range, size_of::<T>()),
            _k: PhantomData,
        }
    }
    #[inline(always)]
    fn initialize(mut ptr: *mut u8, size: usize, len: usize) {
        for _ in 0..len {
            unsafe {
                std::ptr::write(ptr as *mut T, T::null());
                ptr = ptr.add(size);
            }
        }
    }
}

/// An iterator over the elements of a [`Arr<T>`].
///
/// See [`Arr::iter`] for details.
pub struct Iter<'a, T: 'a + Null> {
    raw: RawIter<'a>,
    _k: PhantomData<T>,
}
impl<'a, T: Null> Iter<'a, T> {
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.raw.index()
    }
}
impl<'a, T: Null> Iterator for Iter<'a, T> {
    type Item = &'a mut T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { transmute(self.raw.next()) }
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.raw.size_hint()
    }
}

impl<T: Null> Index<usize> for Arr<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("no element found at index {index}")
    }
}
impl<T: Null> IndexMut<usize> for Arr<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("no element found at index_mut {index}")
    }
}

impl<T: Null> Drop for Arr<T> {
    fn drop(&mut self) {
        for (i, bucket) in self.raw.buckets.iter_mut().enumerate() {
            let entries = *bucket.entries.get_mut() as *mut T;
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
            *arr.get_alloc(i) = value;
        }
        arr
    }
}

impl<T: Null> Extend<T> for Arr<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        for (i, value) in iter.enumerate() {
            *self.get_alloc(i) = value;
        }
    }
}

impl<T: Clone + Null> Clone for Arr<T> {
    fn clone(&self) -> Arr<T> {
        let mut buckets = [ptr::null_mut(); BUCKETS];

        for (i, bucket) in buckets.iter_mut().enumerate() {
            let entries = self.raw.entries(i) as *mut T;
            // bucket is uninitialized
            if entries.is_null() {
                continue;
            }
            let len = Location::bucket_len(i);
            let vec = unsafe { Vec::from_raw_parts(entries, len, len) };
            *bucket = vec.clone().into_raw_parts().0 as *mut u8;
            forget(vec);
        }

        Arr {
            raw: RawArr {
                buckets: buckets.map(Bucket::new),
                lock: ShareMutex::default(),
            },
            _k: PhantomData,
        }
    }
}

impl<T: fmt::Debug + Null> fmt::Debug for Arr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: PartialEq + Null> PartialEq for Arr<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        let mut it = self.iter();
        let mut other = other.iter();
        while let Some(v) = it.next() {
            if let Some(v2) = other.next() {
                if v != v2 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

impl<A, T: Null> PartialEq<A> for Arr<T>
where
    A: AsRef<[T]>,
    T: PartialEq,
{
    fn eq(&self, other: &A) -> bool {
        let other = other.as_ref();

        if self.len() != other.len() {
            return false;
        }
        let mut it = self.iter();
        let mut other = other.iter();
        while let Some(v) = it.next() {
            if let Some(v2) = other.next() {
                if v != v2 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

impl<T: Eq + Null> Eq for Arr<T> {}

#[derive(Default)]
pub struct RawArr {
    // buckets of length 32, 64 .. 2^32
    buckets: [Bucket; BUCKETS],
    lock: ShareMutex<()>,
}

impl RawArr {
    pub fn with_capacity(
        capacity: usize,
        init: fn(*mut u8, usize, usize),
        type_size: usize,
    ) -> RawArr {
        let mut buckets = [ptr::null_mut(); BUCKETS];
        if capacity == 0 {
            return RawArr {
                buckets: buckets.map(Bucket::new),
                lock: ShareMutex::default(),
            };
        }
        let end = Location::of(capacity).bucket;
        for (i, bucket) in buckets[..=end].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = Bucket::alloc(len, init, type_size);
        }

        RawArr {
            buckets: buckets.map(Bucket::new),
            lock: ShareMutex::default(),
        }
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        let mut len = 0;
        for i in 0..self.buckets.len() {
            let entries = self.entries(i);
            // bucket is uninitialized
            if entries.is_null() {
                continue;
            }
            len += Location::bucket_len(i);
        }
        len
    }
    #[inline(always)]
    pub fn get(&self, index: usize, type_size: usize) -> Option<&u8> {
        let location = Location::of(index);

        // safety: `location.bucket` is always in bounds
        let entries = self.entries(location.bucket);

        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }

        // safety: `location.entry` is always in bounds for it's bucket
        Some(unsafe { &*entries.add(location.entry * type_size) })
    }
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize, type_size: usize) -> *mut u8 {
        let location = Location::of(index);

        // safety: caller guarantees the entry is initialized
        self.entries(location.bucket)
            .add(location.entry * type_size)
    }
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize, type_size: usize) -> Option<&mut u8> {
        let location = Location::of(index);
        let entries = self.entries_mut(location.bucket);
        // bucket is uninitialized
        if entries.is_null() {
            return None;
        }
        // safety: `location.entry` is always in bounds for it's bucket
        Some(unsafe { &mut *entries.add(location.entry * type_size) })
    }
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize, type_size: usize) -> *mut u8 {
        let location = Location::of(index);
        self.entries_mut(location.bucket)
            .add(location.entry * type_size)
    }
    #[inline(always)]
    pub unsafe fn get_alloc(
        &mut self,
        index: usize,
        init: fn(*mut u8, usize, usize),
        type_size: usize,
    ) -> *mut u8 {
        let location = Location::of(index);
        let bucket = self.buckets.get_unchecked_mut(location.bucket);
        // safety: `location.bucket` is always in bounds
        let entries = bucket.entries.get_mut();
        // bucket is uninitialized
        if entries.is_null() {
            *entries = Bucket::alloc(location.len, init, type_size);
        }
        // safety: `location.entry` is always in bounds for it's bucket
        entries.add(location.entry * type_size)
    }
    #[inline(always)]
    pub unsafe fn load_alloc(
        &self,
        index: usize,
        init: fn(*mut u8, usize, usize),
        type_size: usize,
    ) -> *mut u8 {
        let location = Location::of(index);
        let bucket = self.buckets.get_unchecked(location.bucket);
        // safety: `location.bucket` is always in bounds
        let mut entries = bucket.entries.load(Ordering::Acquire);
        // bucket is uninitialized
        if entries.is_null() {
            entries = bucket.init(location.len, init, type_size, &self.lock);
        }
        // safety: `location.entry` is always in bounds for it's bucket
        entries.add(location.entry * type_size)
    }
    pub fn replace(&self) -> [(*mut u8, usize); BUCKETS] {
        let mut buckets = [(ptr::null_mut(), 0); BUCKETS];
        let _lock = self.lock.lock();
        for (i, bucket) in self.buckets.iter().enumerate() {
            let e = unsafe { buckets.get_unchecked_mut(i) };
            (*e).0 = bucket.entries.swap(ptr::null_mut(), Ordering::Acquire);
            (*e).1 = Location::bucket_len(i);
        }
        buckets
    }
    pub fn iter(&self, type_size: usize) -> RawIter<'_> {
        self.slice(0..MAX_ENTRIES, type_size)
    }
    pub fn slice(&self, range: Range<usize>, type_size: usize) -> RawIter<'_> {
        let mut start = Location::of(range.start);
        let end = Location::of(range.end);
        if start.bucket == end.bucket {
            start.len = end.entry;
        } else if start.bucket > end.bucket {
            start.len = 0;
        }
        let mut ptr = self.entries(start.bucket);
        if !ptr.is_null() {
            ptr = unsafe { ptr.add(start.entry * type_size) };
        }
        RawIter {
            buckets: &self.buckets,
            type_size,
            start,
            end,
            ptr,
        }
    }
    #[inline(always)]
    fn entries(&self, bucket: usize) -> *mut u8 {
        unsafe {
            self.buckets
                .get_unchecked(bucket)
                .entries
                .load(Ordering::Acquire)
        }
    }
    #[inline(always)]
    fn entries_mut(&mut self, bucket: usize) -> *mut u8 {
        unsafe { *self.buckets.get_unchecked_mut(bucket).entries.get_mut() }
    }
}

#[derive(Default)]
pub struct Bucket {
    entries: SharePtr<u8>,
}

impl Bucket {
    #[inline(always)]
    const fn new(entries: *mut u8) -> Bucket {
        Bucket {
            entries: SharePtr::new(entries),
        }
    }
    fn alloc(len: usize, init: fn(*mut u8, usize, usize), type_size: usize) -> *mut u8 {
        let mut entries = Vec::with_capacity(len * type_size);
        init(entries.as_mut_ptr(), type_size, len);
        entries.into_raw_parts().0
    }
    fn init(
        &self,
        len: usize,
        init: fn(*mut u8, usize, usize),
        type_size: usize,
        lock: &ShareMutex<()>,
    ) -> *mut u8 {
        let _lock = lock.lock();
        let mut ptr = self.entries.load(Ordering::Relaxed);
        if ptr.is_null() {
            ptr = Bucket::alloc(len, init, type_size);
            self.entries.store(ptr, Ordering::Relaxed);
        }
        ptr
    }
}

pub struct RawIter<'a> {
    buckets: &'a [Bucket],
    type_size: usize,
    start: Location,
    end: Location,
    ptr: *mut u8,
}
impl<'a> RawIter<'a> {
    #[inline(always)]
    pub fn empty() -> Self {
        RawIter {
            buckets: &[],
            type_size: 0,
            start: Location::default(),
            end: Location::default(),
            ptr: ptr::null_mut(),
        }
    }
    #[inline(always)]
    pub fn type_size(&self) -> usize {
        self.type_size
    }
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.start.bucket_index()
    }
    #[inline(always)]
    pub(crate) fn step(&mut self) -> &'a mut u8 {
        unsafe {
            let ptr = self.ptr.add(self.type_size);
            transmute(replace(&mut self.ptr, ptr))
        }
    }
}
impl<'a> Iterator for RawIter<'a> {
    type Item = &'a mut u8;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start.entry < self.start.len {
            self.start.entry += 1;
            return Some(self.step());
        }
        loop {
            if self.start.bucket >= self.end.bucket {
                return None;
            }
            self.start.bucket += 1;
            self.ptr = unsafe {
                self.buckets
                    .get_unchecked(self.start.bucket)
                    .entries
                    .load(Ordering::Acquire)
            };
            if !self.ptr.is_null() {
                if self.start.bucket == self.end.bucket {
                    self.start.len = self.end.entry;
                } else {
                    self.start.len = Location::bucket_len(self.start.bucket);
                }
                self.start.entry = 1;
                return Some(self.step());
            } else {
                self.start.len = 0;
            }
        }
    }
    #[inline(always)]
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

#[derive(Debug, Default)]
struct Location {
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
    const fn of(index: usize) -> Location {
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
    fn bucket_index(&self) -> usize {
        ((u32::MAX as u64) >> (u32::BITS - self.bucket as u32) << SKIP_BUCKET) as usize + self.entry
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::*;
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
        arr.insert(98, 98);
        let mut iterator = arr.iter();
        assert_eq!(iterator.size_hint().0, 32);
        let r = iterator.next().unwrap();
        assert_eq!((iterator.index() - 1, *r), (0, 1));
        let r = iterator.next().unwrap();
        assert_eq!((iterator.index() - 1, *r), (1, 2));
        let r = iterator.next().unwrap();
        assert_eq!((iterator.index() - 1, *r), (2, 4));
        for i in 3..32 {
            let r = iterator.next().unwrap();
            assert_eq!((iterator.index() - 1, *r), (i, i32::null()));
        }
        for i in 96..98 {
            let r = iterator.next().unwrap();
            assert_eq!((iterator.index() - 1, *r), (i, i32::null()));
        }
        let r = iterator.next().unwrap();
        assert_eq!((iterator.index() - 1, *r), (98, 98));
        for i in 99..224 {
            let r = iterator.next().unwrap();
            assert_eq!((iterator.index() - 1, *r), (i, i32::null()));
        }
        assert_eq!(iterator.next(), None);
        assert_eq!(iterator.size_hint().0, 0);
    }
    #[test]
    fn test() {
        let arr = arr![1; 3];
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 1);

        let arr = arr![];
        assert_eq!(arr.len(), 0);
        arr.insert(1, 1);
        arr.insert(2, 2);
        assert_eq!(arr.len(), 32);

        let mut arr = arr![10, 40, 30];
        assert_eq!(40, *arr.get_alloc(1));
        assert_eq!(true, arr.get_alloc(3).is_null());
        assert_eq!(40, arr.set(1, 20));
        assert_eq!(true, arr.set(33, 5).is_null());

        {
            let arr: Arr<i8> = arr![10, 40, 30];
            assert_eq!(Some(&40), arr.get(1));
            assert_eq!(None, arr.get(33));
        }

        let arr = crate::arr![1, 2, 4];
        unsafe {
            assert_eq!(arr.get_unchecked(1), &2);
        }

        let mut arr = arr![10, 40, 30];
        assert_eq!(Some(&mut 40), arr.get_mut(1));
        assert_eq!(None, arr.get_mut(33));

        let mut arr = crate::arr![1, 2, 4];
        unsafe {
            assert_eq!(arr.get_unchecked_mut(1), &mut 2);
        }

        let arr = arr![10, 40, 30];
        assert_eq!(40, *arr.load_alloc(1));
        assert_eq!(true, arr.load_alloc(3).is_null());
        assert_eq!(true, arr.load_alloc(133).is_null());

        let arr = arr![10, 40, 30];
        assert_eq!(40, *arr.load(1).unwrap());
        assert_eq!(true, arr.load(3).unwrap().is_null());
        assert_eq!(None, arr.load(133));

        let arr = crate::arr![1, 2];
        arr.insert(2, 3);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);

        let arr = crate::arr![1, 2, 4];
        arr.insert(98, 98);
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
                    arr.insert(i, i);
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
                arr.insert(i, Some(Arc::new(i)));
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
        arr.insert(0, Some(Mutex::new(1)));

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
            assert_eq!(loc.bucket_index(), 0)
        }

        assert_eq!(Location::bucket_len(1), 64);
        for i in 33..96 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 64);
            assert_eq!(loc.bucket, 1);
            assert_eq!(loc.entry, i - 32);
            assert_eq!(loc.bucket_index(), 32)
        }

        assert_eq!(Location::bucket_len(2), 128);
        for i in 96..224 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 128);
            assert_eq!(loc.bucket, 2);
            assert_eq!(loc.entry, i - 96);
            assert_eq!(loc.bucket_index(), 96)
        }

        let max = Location::of(MAX_ENTRIES);
        assert_eq!(max.bucket, BUCKETS - 1);
        assert_eq!(max.len, 1 << 31);
        assert_eq!(max.entry, (1 << 31) - 1);
    }
}
