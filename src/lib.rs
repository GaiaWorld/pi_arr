use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range};
use std::sync::atomic::Ordering;
use std::{ptr, slice};

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
/// let arr = arr![1, 2, 3];
/// assert_eq!(arr[0], 1);
/// assert_eq!(arr[1], 2);
/// assert_eq!(arr[2], 3);
/// ```
///
/// - Create a [`Arr`] from a given element and size:
///
/// ```
/// let arr = arr![1; 3];
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

/// This entry is a raw pointer internally and can be saved externally.
///  When calling, you must ensure that the container is not free
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Entry<T>(*mut T);
impl<T> Null for Entry<T> {
    fn null() -> Self {
        Entry(ptr::null_mut())
    }
    fn is_null(&self) -> bool {
        self.0.is_null()
    }
}
impl<T> AsRef<T> for Entry<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        unsafe { &*self.0 }
    }
}
impl<T> AsMut<T> for Entry<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}
impl<T> Deref for Entry<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}
impl<T> DerefMut for Entry<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}
impl<T> Default for Entry<T> {
    #[inline]
    fn default() -> Entry<T> {
        Entry(ptr::null_mut())
    }
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
    // buckets of length 32, 64 .. 2^32
    buckets: [Bucket<T>; BUCKETS],
    lock: ShareMutex<()>,
}

unsafe impl<T: Send + Null> Send for Arr<T> {}
unsafe impl<T: Sync + Null> Sync for Arr<T> {}

impl<T: Null> Arr<T> {
    /// Constructs a new, empty `Arr<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr: pi_arr::arr<i32> = pi_arr::arr::new();
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
    /// let arr = pi_arr::arr::with_capacity(10);
    ///
    /// for i in 0..32 {
    ///     // will not allocate
    ///     arr.set(i, i);
    /// }
    ///
    /// // may allocate
    /// arr.set(33, 33);
    /// ```
    pub fn with_capacity(capacity: usize) -> Arr<T> {
        let mut buckets = [ptr::null_mut(); BUCKETS];
        if capacity == 0 {
            return Arr {
                buckets: buckets.map(Bucket::new),
                lock: ShareMutex::default(),
            };
        }
        let init = Location::of(capacity).bucket;
        for (i, bucket) in buckets[..=init].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = Bucket::alloc(len);
        }

        Arr {
            buckets: buckets.map(Bucket::new),
            lock: ShareMutex::default(),
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
    /// let arr = pi_arr::arr::new();
    /// assert_eq!(arr.len(), 0);
    /// arr.set(1, 1);
    /// arr.set(2, 2);
    /// assert_eq!(arr.len(), 32);
    /// ```
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

    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&40), arr.get(1));
    /// assert_eq!(None, arr.get(33));
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        let location = Location::of(index);

        // safety: `location.bucket` is always in bounds
        let entries = self.entries(location.bucket);

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
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        let location = Location::of(index);

        // safety: caller guarantees the entry is initialized
        &*self.entries(location.bucket).add(location.entry)
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
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let location = Location::of(index);

        // safety: `location.bucket` is always in bounds
        let entries = unsafe {
            self.buckets
                .get_unchecked_mut(location.bucket)
                .entries
                .get_mut()
        };

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
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        let location = Location::of(index);

        // safety: caller guarantees the entry is initialized
        &mut *self
            .buckets
            .get_unchecked_mut(location.bucket)
            .entries
            .get_mut()
            .add(location.entry)
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&mut 40), arr.mut_index(1));
    /// assert_eq!(None, arr.mut_index(33));
    /// ```
    pub fn mut_index(&self, index: usize) -> Option<&mut T> {
        let location = Location::of(index);

        // safety: `location.bucket` is always in bounds
        let entries = self.entries(location.bucket);

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
    ///     assert_eq!(arr.unchecked_mut_index(1), &mut 2);
    /// }
    /// ```
    pub unsafe fn unchecked_mut_index(&self, index: usize) -> &mut T {
        let location = Location::of(index);
        // safety: caller guarantees the entry is initialized
        &mut *self.entries(location.bucket).add(location.entry)
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned entry is default
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.entry(1).as_ref());
    /// assert_eq!(0, *arr.entry(3).as_ref());
    /// ```
    pub fn entry(&self, index: usize) -> Entry<T> {
        let location = Location::of(index);
        let bucket = unsafe { self.buckets.get_unchecked(location.bucket) };
        // safety: `location.bucket` is always in bounds
        let mut entries = bucket.entries.load(Ordering::Acquire);
        // bucket is uninitialized
        if entries.is_null() {
            entries = bucket.init(location.bucket_len, &self.lock);
        }
        // safety: `location.entry` is always in bounds for it's bucket
        unsafe { Entry(entries.add(location.entry)) }
    }
    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will not be allocated automatically, and the returned entry is null
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.try_entry(1).as_ref());
    /// assert_eq!(true, arr.try_entry(33).is_null());
    /// ```
    pub fn try_entry(&self, index: usize) -> Entry<T> {
        let location = Location::of(index);
        // safety: `location.bucket` is always in bounds
        let entries = self.entries(location.bucket);
        // bucket is uninitialized
        if entries.is_null() {
            return Entry(ptr::null_mut());
        }
        // safety: `location.entry` is always in bounds for it's bucket
        unsafe { Entry(entries.add(location.entry)) }
    }
    /// set an element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2];
    /// arr.set(2, 3);
    /// assert_eq!(arr[0], 1);
    /// assert_eq!(arr[1], 2);
    /// assert_eq!(arr[2], 3);
    /// ```
    #[inline]
    pub fn set(&self, index: usize, value: T) {
        *self.entry(index).as_mut() = value;
    }
    /// clear and free buckets.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = pi_arr::arr![1, 2];
    /// arr.clear();
    /// arr.set(2, 3);
    /// assert_eq!(arr[0], 0);
    /// assert_eq!(arr[1], 0);
    /// assert_eq!(arr[2], 3);
    /// ```
    pub fn clear(&self) {
        for (i, bucket) in self.buckets.iter().enumerate() {
            let entries = bucket.entries.load(Ordering::Acquire);
            if entries.is_null() {
                continue;
            }
            let len = Location::bucket_len(i);
            // safety: in drop
            unsafe { drop(Box::from_raw(slice::from_raw_parts_mut(entries, len))) }
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
    /// let arr = pi_arr::arr![1, 2, 4];
    /// let mut iterator = arr.iter();
    ///
    /// let r = iterator.next().unwrap();
    /// assert_eq!((r.0, *r.1), (0, 1));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((r.0, *r.1), (1, 2));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((r.0, *r.1), (2, 4));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((r.0, *r.1), (3, 0));
    /// assert_eq!(iterator.size_hint().0, 32-4);
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
    /// assert_eq!((r.0, *r.1), (1, 2));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((r.0, *r.1), (2, 4));
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn slice(&self, range: Range<usize>) -> Iter<'_, T> {
        let mut start = Location::of(range.start);
        let mut end = Location::of(range.end);
        let mut cur = ptr::null_mut();
        while start.bucket <= end.bucket {
            cur = self.entries(start.bucket);
            if !cur.is_null() {
                break;
            }
            start.up();
        }
        while end.bucket > start.bucket {
            let entries = self.entries(end.bucket);
            if !entries.is_null() {
                break;
            }
            end.down();
        }
        let size = if start.bucket < end.bucket {
            let mut size = end.entry + start.bucket_len - start.entry;
            for bucket in (start.bucket + 1)..end.bucket {
                let entries = self.entries(bucket);
                if !entries.is_null() {
                    size += Location::bucket_len(bucket);
                }
            }
            size
        } else if end.bucket == end.bucket {
            end.entry.checked_sub(start.entry).unwrap_or(0)
        } else {
            0
        };
        Iter {
            arr: &self,
            start,
            end,
            cur,
            size,
        }
    }
    #[inline]
    fn entries(&self, bucket: usize) -> *mut T {
        unsafe {
            self.buckets
                .get_unchecked(bucket)
                .entries
                .load(Ordering::Acquire)
        }
    }
}

impl<T: Null> Index<usize> for Arr<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("no element found at index {index}")
    }
}
impl<T: Null> IndexMut<usize> for Arr<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("no element found at index_mut {index}")
    }
}

impl<T: Null> Drop for Arr<T> {
    fn drop(&mut self) {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let entries = *bucket.entries.get_mut();
            if entries.is_null() {
                continue;
            }
            let len = Location::bucket_len(i);
            // safety: in drop
            unsafe { drop(Box::from_raw(slice::from_raw_parts_mut(entries, len))) }
        }
    }
}

impl<T: Null> FromIterator<T> for Arr<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let mut arr = Arr::with_capacity(lower);
        for (i, value) in iter.enumerate() {
            match arr.get_mut(i) {
                Some(x) => *x = value,
                None => arr.set(i, value),
            }
        }
        arr
    }
}

impl<T: Null> Extend<T> for Arr<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        for (i, value) in iter.enumerate() {
            match self.get_mut(i) {
                Some(x) => *x = value,
                None => self.set(i, value),
            }
        }
    }
}

impl<T: Clone + Null> Clone for Arr<T> {
    fn clone(&self) -> Arr<T> {
        let mut buckets = [ptr::null_mut(); BUCKETS];

        for (i, bucket) in buckets.iter_mut().enumerate() {
            let entries = self.entries(i);

            // bucket is uninitialized
            if entries.is_null() {
                continue;
            }
            let len = Location::bucket_len(i);
            *bucket = Bucket::alloc(len);
            unsafe {
                ptr::copy(entries, *bucket, len);
            }
        }

        Arr {
            buckets: buckets.map(Bucket::new),
            lock: ShareMutex::default(),
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

        // ensure indexes are checked along with values to handle gaps in the array
        for (index, value) in self.iter() {
            if other.try_entry(index) != value {
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

        // ensure indexes are checked along with values to handle gaps in the array
        for (index, value) in self.iter() {
            if other.get(index) != Some(value.as_ref()) {
                return false;
            }
        }

        true
    }
}

impl<T: Eq + Null> Eq for Arr<T> {}

/// An iterator over the elements of a [`Arr<T>`].
///
/// See [`Arr::iter`] for details.
pub struct Iter<'a, T: Null> {
    arr: &'a Arr<T>,
    start: Location,
    end: Location,
    cur: *mut T,
    size: usize,
}
impl<'a, T: Null> Iterator for Iter<'a, T> {
    type Item = (usize, Entry<T>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.size == 0 {
            return None;
        }
        let r = unsafe { (self.start.entry, Entry(self.cur.add(self.start.entry))) };
        self.size -= 1;
        self.start.entry += 1;
        if self.start.entry < self.start.bucket_len {
            return Some(r);
        }
        self.start.up();
        while self.start.bucket <= self.end.bucket {
            self.cur = self.arr.entries(self.start.bucket);
            if !self.cur.is_null() {
                break;
            }
            self.start.up();
        }
        Some(r)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

#[derive(Default)]
struct Bucket<T: Null> {
    entries: SharePtr<T>,
}

impl<T: Null> Bucket<T> {
    fn new(entries: *mut T) -> Bucket<T> {
        Bucket {
            entries: SharePtr::new(entries),
        }
    }
    fn alloc(len: usize) -> *mut T {
        let entries = (0..len).map(|_| T::null()).collect::<Box<[T]>>();
        Box::into_raw(entries) as _
    }
    fn init(&self, len: usize, lock: &ShareMutex<()>) -> *mut T {
        let _lock = lock.lock();
        let mut ptr = self.entries.load(Ordering::Acquire);
        if ptr.is_null() {
            ptr = Bucket::alloc(len);
            self.entries.store(ptr, Ordering::Release);
        }
        ptr
    }
}

#[derive(Debug)]
struct Location {
    // the index of the bucket
    bucket: usize,
    // the length of `bucket`
    bucket_len: usize,
    // the index of the entry in `bucket`
    entry: usize,
}

// skip the shorter buckets to avoid unnecessary allocations.
// this also reduces the maximum capacity of a arr.
const SKIP: usize = 32;
const SKIP_BUCKET: usize = ((usize::BITS - SKIP.leading_zeros()) as usize) - 1;

impl Location {
    fn of(index: usize) -> Location {
        let skipped = index.checked_add(SKIP).expect("exceeded maximum length");
        let bucket = usize::BITS - skipped.leading_zeros();
        let bucket = (bucket as usize) - (SKIP_BUCKET + 1);
        let bucket_len = Location::bucket_len(bucket);
        let entry = skipped ^ bucket_len;

        Location {
            bucket,
            bucket_len,
            entry,
        }
    }
    #[inline]
    fn bucket_len(bucket: usize) -> usize {
        1 << (bucket + SKIP_BUCKET)
    }
    fn up(&mut self) {
        self.bucket += 1;
        self.bucket_len = Location::bucket_len(self.bucket);
        self.entry = 0;
    }
    fn down(&mut self) {
        self.bucket -= 1;
        self.bucket_len = Location::bucket_len(self.bucket);
        self.entry = self.bucket_len;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::*;

    #[test]
    fn test() {
        let arr = arr![1; 3];
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 1);

        let arr = arr![];
        assert_eq!(arr.len(), 0);
        arr.set(1, 1);
        arr.set(2, 2);
        assert_eq!(arr.len(), 32);

        let arr = arr![10, 40, 30];
        assert_eq!(Some(&40), arr.get(1));
        assert_eq!(None, arr.get(33));

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
        assert_eq!(40, *arr.entry(1).as_ref());
        assert_eq!(0, *arr.entry(3).as_ref());

        let arr = arr![10, 40, 30];
        assert_eq!(40, *arr.try_entry(1).as_ref());
        assert_eq!(true, arr.try_entry(33).is_null());

        let arr = crate::arr![1, 2];
        arr.set(2, 3);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);

        let arr = crate::arr![1, 2, 4];
        let mut iterator = arr.iter();
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (0, 1));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (1, 2));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (2, 4));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (3, 0));
        assert_eq!(iterator.size_hint().0, 32 - 4);

        let mut iterator = arr.slice(1..3);
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (1, 2));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (2, 4));
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
                    arr.set(i, i);
                })
            })
            .collect::<Vec<_>>();

        // wait for the threads to finish
        for thread in threads {
            thread.join().unwrap();
        }

        for i in 0..6 {
            assert!(arr.iter().any(|(_, x)| x.as_ref() == &i));
        }
    }
    #[test]
    fn test_mutex() {
        let arr = Arc::new(crate::Arr::new());

        // insert an element
        arr.set(0, Some(Mutex::new(1)));

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
            assert_eq!(loc.bucket_len, 32);
            assert_eq!(loc.bucket, 0);
            assert_eq!(loc.entry, i);
        }

        assert_eq!(Location::bucket_len(1), 64);
        for i in 33..96 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 64);
            assert_eq!(loc.bucket, 1);
            assert_eq!(loc.entry, i - 32);
        }

        assert_eq!(Location::bucket_len(2), 128);
        for i in 96..224 {
            let loc = Location::of(i);
            assert_eq!(loc.bucket_len, 128);
            assert_eq!(loc.bucket, 2);
            assert_eq!(loc.entry, i - 96);
        }

        let max = Location::of(MAX_ENTRIES);
        assert_eq!(max.bucket, BUCKETS - 1);
        assert_eq!(max.bucket_len, 1 << 31);
        assert_eq!(max.entry, (1 << 31) - 1);
    }
}
