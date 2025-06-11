//! 自动扩展的数组VecArr，由一个可自动扩容的Vec实现。仅在wasm环境下使用的。
//! 自动扩展的数组VBArr。在多线程环境下使用的。但各种访问方法，要求外部保证，多线程调用时不要同时访问到同一个元素，否则会引发数据竞争。
//! 由一个主数组(可扩展)和多个固定大小的辅助数组构成。
//! 当主数组上的长度不够时，不会立刻扩容，而是线程安全的在辅助数组分配新辅助数组。
//! 在独占整理时，会合并所有辅助数组上的数据到主数组。


#![feature(unsafe_cell_access)]
#![feature(vec_into_raw_parts)]
#![feature(test)]
extern crate test;

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::{forget, replace, size_of, transmute};
use std::ops::{Index, IndexMut, Range};
use std::ptr::{null_mut, NonNull};

#[cfg(not(feature = "rc"))]
use pi_buckets::{bucket_alloc, BucketIter, Buckets, Location, SKIP};

use pi_vec_remain::VecRemain;

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

#[cfg(feature = "rc")]
pub type Arr<T> = VecArr<T>;

#[cfg(feature = "rc")]
pub type Iter<'a, T> = VecIter<'a, T>;

#[cfg(not(feature = "rc"))]
pub type Arr<T> = VBArr<T>;

#[cfg(not(feature = "rc"))]
pub type Iter<'a, T> = BucketIter<'a, T>;

/// A lock-free, auto-expansion array by buckets.
///
/// See [the crate documentation](crate) for details.
///
/// # Notes
///
/// The bucket array is stored inline, meaning that the
/// `Arr<T>` is quite large. It is expected that you
/// store it behind an [`Arc`](std::sync::Arc) or similar.
pub struct VBArr<T> {
    ptr: *mut T,
    capacity: usize,
    pub buckets: *mut Buckets<T>,
}
#[cfg(not(feature = "rc"))]
impl<T> Default for VBArr<T> {
    fn default() -> Self {
        if size_of::<T>() == 0 {
            return VBArr {
                ptr: NonNull::<T>::dangling().as_ptr(),
                capacity: usize::MAX,
                buckets: NonNull::<Buckets<T>>::dangling().as_ptr(),
            };
        }
        let ptr = NonNull::<T>::dangling().as_ptr();
        let buckets = Box::into_raw(Box::new(Default::default()));
        VBArr {
            ptr,
            capacity: 0,
            buckets,
        }
    }
}
unsafe impl<T: Send> Send for Arr<T> {}
unsafe impl<T: Sync> Sync for Arr<T> {}

#[cfg(not(feature = "rc"))]
impl<T: Default> VBArr<T> {
    /// Constructs a new, empty `Arr<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
    /// let arr: pi_arr::Arr<i32> = pi_arr::Arr::new();
    /// ```
    #[inline]
    pub fn new() -> VBArr<T> {
        VBArr::default()
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
    /// use crate::pi_arr;
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
    pub fn with_capacity(capacity: usize) -> VBArr<T> {
        if size_of::<T>() == 0 {
            return VBArr {
                ptr: NonNull::<T>::dangling().as_ptr(),
                capacity: usize::MAX,
                buckets: NonNull::<Buckets<T>>::dangling().as_ptr(),
            };
        }
        let ptr = if capacity == 0 {
            NonNull::<T>::dangling().as_ptr()
        } else {
            let vec = Vec::with_capacity(capacity);
            vec.into_raw_parts().0
        };
        let buckets = Box::into_raw(Box::new(Default::default()));
        VBArr {
            ptr,
            capacity,
            buckets,
        }
    }
    /// 获得容量大小
    #[inline(always)]
    pub fn capacity(&self, len: usize) -> usize {
        if size_of::<T>() == 0 {
            0
        } else if len > self.capacity {
            Location::bucket_capacity(Location::bucket(len - self.capacity)) + self.capacity
        } else {
            self.capacity
        }
    }

    #[inline(always)]
    pub fn vec_capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.capacity
        }
    }
    #[inline(always)]
    fn buckets(&self) -> &Buckets<T> {
        unsafe { &*self.buckets }
    }
    #[inline(always)]
    fn buckets_mut(&mut self) -> &mut Buckets<T> {
        unsafe { &mut *self.buckets }
    }
    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&40), arr.get(1));
    /// assert_eq!(None, arr.get(33));
    /// ```
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &*self.ptr.add(index) });
        }
        self.buckets().get(&Location::of(index - self.capacity))
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
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.get_unchecked(1), &2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        if index < self.vec_capacity() {
            return unsafe { &*self.ptr.add(index) };
        }
        self.buckets()
            .get_unchecked(&Location::of(index - self.capacity))
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&mut 40), arr.get_mut(1));
    /// assert_eq!(None, arr.get_mut(33));
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &mut *self.ptr.add(index) });
        }
        let loc = &Location::of(index - self.capacity);
        self.buckets_mut().get_mut(loc)
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
    /// use crate::pi_arr;
    /// let mut arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.get_unchecked_mut(1), &mut 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        if index < self.vec_capacity() {
            return unsafe { &mut *self.ptr.add(index) };
        }
        let loc = &Location::of(index - self.capacity);
        self.buckets_mut().get_unchecked_mut(loc)
    }
    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is default value.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.alloc(1));
    /// assert_eq!(0, *arr.alloc(3));
    /// ```
    #[inline(always)]
    pub fn alloc(&mut self, index: usize) -> &mut T {
        if index < self.vec_capacity() {
            return unsafe { &mut *self.ptr.add(index) };
        }
        let loc = &Location::of(index - self.capacity);
        self.buckets_mut().alloc(loc)
    }
    /// set element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let mut arr = crate::pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, arr.set(1, 20));
    /// assert_eq!(Some(&20), arr.get(1));
    /// assert_eq!(0, arr.set(33, 5));
    /// assert_eq!(Some(&5), arr.get(33));
    /// ```
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: T) -> T {
        replace(self.alloc(index), value)
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will not be allocated automatically, and the returned None.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(10, *arr.load(0).unwrap());
    /// assert_eq!(Some(&mut 40), arr.load(1));
    /// assert_eq!(None, arr.load(3));
    /// assert_eq!(None, arr.load(33));
    /// ```
    #[inline(always)]
    pub fn load(&self, index: usize) -> Option<&mut T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &mut *self.ptr.add(index) });
        }
        self.buckets().load(&Location::of(index - self.capacity))
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
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.load_unchecked(1), &mut 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn load_unchecked(&self, index: usize) -> &mut T {
        if index < self.vec_capacity() {
            return unsafe { &mut *self.ptr.add(index) };
        }
        self.buckets()
            .load_unchecked(&Location::of(index - self.capacity))
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is default value.
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.load_alloc(1));
    /// assert_eq!(0, *arr.load_alloc(3));
    /// ```
    #[inline(always)]
    pub fn load_alloc(&self, index: usize) -> &mut T {
        if index < self.vec_capacity() {
            return unsafe { &mut *self.ptr.add(index) };
        }
        self.buckets()
            .load_alloc(&Location::of(index - self.capacity))
    }

    /// insert an element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
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

    /// Returns an iterator over the array at the given range.
    ///
    /// Values are yielded in the form `Entry`.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![1, 2, 4];
    /// let mut iterator = arr.slice(0..4294967263);
    /// assert_eq!(iterator.size_hint().0, 3);
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (0, 1));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (1, 2));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (2, 4));
    /// assert_eq!(iterator.next(), None);
    /// assert_eq!(iterator.size_hint().0, 0);
    ///
    /// let arr = crate::pi_arr::arr![1, 2, 4, 6];
    /// let mut iterator = arr.slice(1..3);
    ///
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 2);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 4);
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> BucketIter<'_, T> {
        if range.end <= self.vec_capacity() {
            BucketIter::new(
                self.ptr,
                Location::new(-1, range.end, range.start),
                range.end,
                -1,
                &self.buckets(),
                self.capacity,
            )
        } else if range.start < self.capacity {
            let end = Location::of(range.end - self.capacity);
            BucketIter::new(
                self.ptr,
                Location::new(-1, self.capacity, range.start),
                end.entry,
                end.bucket,
                &self.buckets(),
                self.capacity,
            )
        } else {
            self.buckets().slice_row(range, self.capacity)
        }
    }
    /// 保留范围内的数组元素并整理，将保留的部分整理到扩展槽中，并将当前vec_capacity容量扩容len+additional
    pub fn remain_settle(&mut self, range: Range<usize>, len: usize, additional: usize) {
        if size_of::<T>() == 0 {
            return;
        }
        debug_assert!(len >= range.end);
        debug_assert!(range.end >= range.start);
        let mut vec = to_vec(self.ptr, self.capacity);
        if range.end <= self.capacity {
            // 数据都在vec上
            vec.remain(range.start..range.end);
            if len > self.capacity {
                self.buckets().take();
            }
            return self.reserve(vec, range.len(), additional);
        }
        // 取出所有的bucket
        let mut arr = self.buckets().take();
        // 获得最后一个bucket的索引
        let bucket_end = Location::bucket(range.end - self.capacity);
        if vec.capacity() == 0 && range.start == 0 && bucket_end == 0 {
            // 如果vec为空，且范围从0开始，且bucket_end为0，则表示只保留第一个槽的数据，将vec交换成第一个槽的数据
            vec = replace(&mut arr[0], Vec::new());
            // 保留范围内的数据
            vec.remain(range.start..range.end);
            self.capacity = vec.capacity();
            self.ptr = vec.into_raw_parts().0;
            return;
        }
        // 总是尝试在vec上保留范围内的数据
        vec.remain(range.start..range.end);
        // 获得扩容后的总容量
        let cap = range.len() + additional;
        // println!("vec.capacity() = {}, cap = {}", vec.capacity(), cap);
        // 如果vec容量小于cap，则将vec容量扩展到cap
        if vec.capacity() < cap {
            vec.reserve(cap - vec.capacity());
        }
        // 获得第一个bucket的索引
         let bucket_start = Location::bucket(range.start.saturating_sub(self.capacity));
        let mut start = self.capacity + Location::bucket_len(bucket_start) - SKIP;
        // println!("bucket_start_end = {:?}, start = {}", (bucket_start..bucket_end), start);
        // 将arr的数据拷贝到vec上
        for (i, v) in arr[bucket_start..bucket_end + 1].iter_mut().enumerate() {
            let mut vlen = v.len();
            // println!("i = {:?}, vlen = {}", i, vlen);
            if vlen > 0 {
                v.remain_to(range.start.saturating_sub(start)..range.end - start, &mut vec);
            } else {
                vlen = Location::bucket_len(bucket_start + i);
                let left = range.start.saturating_sub(start);
                let right = (range.end - start).min(vlen);
                // println!(" ========== {:?}", (vlen, left, right));
                vec.resize_with(vec.len() + right - left, || T::default());
            }
            start += vlen;
        }
        // 如果容量比len大，则初始化为缺省值
        vec.resize_with(vec.capacity(), || T::default());
        self.capacity = vec.capacity();
        self.ptr = vec.into_raw_parts().0;
    }
    /// 整理内存，将bucket_arr的数据移到vec上，并将当前vec_capacity容量扩容len+additional
    pub fn settle(&mut self, len: usize, additional: usize) {
        self.remain_settle(0..len, len, additional);
    }
    /// 清理所有数据，释放内存，并尝试将当前vec_capacity容量扩容len+additional
    /// 释放前，必须先调用clear方法，保证释放其中的数据
    #[inline(always)]
    pub fn clear(&mut self, len: usize, additional: usize) {
        if size_of::<T>() == 0 {
            return;
        }
        if len > self.capacity {
            self.buckets().take();
        }
        let mut vec = to_vec(self.ptr, self.capacity);
        vec.clear();
        self.reserve(vec, len, additional);
    }
    fn reserve(&mut self, mut vec: Vec<T>, len: usize, mut additional: usize) {
        additional = (len + additional).saturating_sub(self.capacity);
        if additional > 0 {
            vec.reserve(additional);
            vec.resize_with(vec.capacity(), || T::default());
            self.capacity = vec.capacity();
        }
        self.ptr = vec.into_raw_parts().0;
    }
}

impl<T: Default> Index<usize> for Arr<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("no element found at index {index}")
    }
}
impl<T: Default> IndexMut<usize> for Arr<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("no element found at index_mut {index}")
    }
}

impl<T> Drop for VBArr<T> {
    fn drop(&mut self) {
        if size_of::<T>() == 0 {
            return;
        }
        to_vec(self.ptr, self.capacity);
        unsafe { drop(Box::from_raw(self.buckets)) };
    }
}
#[cfg(not(feature = "rc"))]
impl<T: Default> FromIterator<T> for VBArr<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let mut arr = VBArr::with_capacity(lower);
        for (i, value) in iter.enumerate() {
            arr.set(i, value);
        }
        arr
    }
}
#[cfg(not(feature = "rc"))]
impl<T: Default> Extend<T> for VBArr<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        for (i, value) in iter.enumerate() {
            self.set(i, value);
        }
    }
}
#[cfg(not(feature = "rc"))]
impl<T: Default + Clone> Clone for VBArr<T> {
    fn clone(&self) -> VBArr<T> {
        if size_of::<T>() == 0 {
            return VBArr::default();
        }
        let vec = to_vec(self.ptr, self.capacity);
        let ptr = vec.clone().into_raw_parts().0;
        forget(vec);
        let buckets = Box::into_raw(Box::new(self.buckets().clone()));
        VBArr {
            ptr,
            capacity: self.capacity,
            buckets,
        }
    }
}

pub struct VecArr<T> {
    ptr: UnsafeCell<*mut T>,
    capacity: UnsafeCell<usize>,
}
impl<T> Default for VecArr<T> {
    fn default() -> Self {
        if size_of::<T>() == 0 {
            return Self {
                ptr: NonNull::<T>::dangling().as_ptr().into(),
                capacity: usize::MAX.into(),
            };
        }
        Self {
            ptr: NonNull::<T>::dangling().as_ptr().into(),
            capacity: 0.into(),
        }
    }
}

impl<T: Default> VecArr<T> {
    pub fn new() -> VecArr<T> {
        VecArr::default()
    }
    pub fn with_capacity(capacity: usize) -> VecArr<T> {
        if size_of::<T>() == 0 {
            return Self {
                ptr: NonNull::<T>::dangling().as_ptr().into(),
                capacity: usize::MAX.into(),
            };
        }
        let ptr = if capacity == 0 {
            NonNull::<T>::dangling().as_ptr().into()
        } else {
            bucket_alloc::<T>(capacity).into()
        };
        VecArr {
            ptr,
            capacity: capacity.into(),
        }
    }
    /// 获得容量大小
    #[inline(always)]
    pub fn capacity(&self, _len: usize) -> usize {
        if size_of::<T>() == 0 {
            0
        } else {
            *unsafe { self.capacity.as_ref_unchecked() }
        }
    }
    #[inline(always)]
    pub unsafe fn set_vec_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.into();
    }
    #[inline(always)]
    pub fn vec_capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            *unsafe { self.capacity.as_ref_unchecked() }
        }
    }
    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&40), arr.get(1));
    /// assert_eq!(None, arr.get(33));
    /// ```
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &*(*self.ptr.get()).add(index) });
        }
        None
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
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.get_unchecked(1), &2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        return unsafe { &*(*self.ptr.get()).add(index) };
    }

    /// Returns a mutable reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(Some(&mut 40), arr.get_mut(1));
    /// assert_eq!(None, arr.get_mut(33));
    /// ```
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &mut *(*self.ptr.get()).add(index) });
        }
        None
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
    /// use crate::pi_arr;
    /// let mut arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.get_unchecked_mut(1), &mut 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        return unsafe { &mut *(*self.ptr.get()).add(index) };
    }
    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is defalt value.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let mut arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.alloc(1));
    /// assert_eq!(0, *arr.alloc(3));
    /// ```
    #[inline(always)]
    pub fn alloc(&mut self, index: usize) -> &mut T {
        if index >= self.vec_capacity() {
            let vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity());
            self.reserve(vec, self.vec_capacity(), index - self.vec_capacity() + 1);
        }
        return unsafe { &mut *(*self.ptr.get()).add(index) };
    }
    /// set element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let mut arr = crate::pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, arr.set(1, 20));
    /// assert_eq!(Some(&20), arr.get(1));
    /// assert_eq!(0, arr.set(33, 5));
    /// assert_eq!(Some(&5), arr.get(33));
    /// ```
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: T) -> T {
        replace(self.alloc(index), value)
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will not be allocated automatically, and the returned None.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(10, *arr.load(0).unwrap());
    /// assert_eq!(Some(&mut 40), arr.load(1));
    /// assert_eq!(None, arr.load(3));
    /// assert_eq!(None, arr.load(33));
    /// ```
    #[inline(always)]
    pub fn load(&self, index: usize) -> Option<&mut T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &mut *(*self.ptr.get()).add(index) });
        }
        None
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
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.load_unchecked(1), &mut 2);
    /// }
    /// ```
    #[inline(always)]
    pub unsafe fn load_unchecked(&self, index: usize) -> &mut T {
        return unsafe { &mut *(*self.ptr.get()).add(index) };
    }

    /// Returns a mutable reference to the element at the given index.
    /// If the bucket corresponding to the index is not allocated,
    /// it will be allocated automatically, and the returned T is default value.
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![10, 40, 30];
    /// assert_eq!(40, *arr.load_alloc(1));
    /// assert_eq!(0, *arr.load_alloc(3));
    /// ```
    #[inline(always)]
    pub fn load_alloc(&self, index: usize) -> &mut T {
        if index >= self.vec_capacity() {
            let vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity());
            self.reserve(vec, self.vec_capacity(), index - self.vec_capacity() + 1);
        }
        return unsafe { &mut *(*self.ptr.get()).add(index) };
    }

    /// insert an element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
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

    /// Returns an iterator over the array at the given range.
    ///
    /// Values are yielded in the form `Entry`.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use crate::pi_arr;
    /// let arr = pi_arr::arr![1, 2, 4];
    /// let mut iterator = arr.slice(0..4294967263);
    /// assert_eq!(iterator.size_hint().0, 3);
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (0, 1));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (1, 2));
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (2, 4));
    /// assert_eq!(iterator.next(), None);
    /// assert_eq!(iterator.size_hint().0, 0);
    ///
    /// let arr = crate::pi_arr::arr![1, 2, 4, 6];
    /// let mut iterator = arr.slice(1..3);
    ///
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 2);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 4);
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> VecIter<'_, T> {
        if range.end <= self.vec_capacity() {
            VecIter::new(unsafe { *self.ptr.get() }, range.start, range.end)
        } else if range.start < self.vec_capacity() {
            VecIter::new(unsafe { *self.ptr.get() }, range.start, self.vec_capacity())
        } else {
            VecIter::empty()
        }
    }
    /// 保留范围内的数组元素并整理，将保留的部分整理到扩展槽中，并将当前vec_capacity容量扩容len+additional
    pub fn remain_settle(&mut self, range: Range<usize>, _len: usize, additional: usize) {
        if size_of::<T>() == 0 {
            return;
        }
        let mut vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity());
        // 数据都在vec上
        vec.remain(range.start..range.end);
        return self.reserve(vec, range.len(), additional);
    }
    /// 整理内存，将bucket_arr的数据移到vec上，并将当前vec_capacity容量扩容len+additional
    pub fn settle(&mut self, _len: usize, _additional: usize) {}
    /// 清理所有数据，释放bucket_arr的内存，并尝试将当前vec_capacity容量扩容len+additional
    /// 释放前，必须先调用clear方法，保证释放其中的数据
    #[inline(always)]
    pub fn clear(&mut self, len: usize, additional: usize) {
        if size_of::<T>() == 0 {
            return;
        }
        let mut vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity());
        vec.clear();
        self.reserve(vec, len, additional);
    }
    fn reserve(&self, mut vec: Vec<T>, len: usize, mut additional: usize) {
        additional = (len + additional).saturating_sub(self.vec_capacity());
        if additional > 0 {
            vec.reserve(additional);
            vec.resize_with(vec.capacity(), || T::default());
            unsafe { self.capacity.replace(vec.capacity()) };
        }
        unsafe { self.ptr.replace(vec.into_raw_parts().0) };
    }
}
impl<T> Drop for VecArr<T> {
    fn drop(&mut self) {
        if size_of::<T>() == 0 {
            return;
        }
        let len = unsafe { *self.capacity.as_ref_unchecked() };
        if len == usize::MAX {
            return;
        }
        to_vec(unsafe { *self.ptr.get() }, len);
    }
}

impl<T: Default> FromIterator<T> for VecArr<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let mut arr = VecArr::with_capacity(lower);
        for (i, value) in iter.enumerate() {
            arr.set(i, value);
        }
        arr
    }
}

impl<T: Default> Extend<T> for VecArr<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        for (i, value) in iter.enumerate() {
            self.set(i, value);
        }
    }
}

impl<T: Default + Clone> Clone for VecArr<T> {
    fn clone(&self) -> Self {
        if size_of::<T>() == 0 {
            return VecArr::default();
        }
        let vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity());
        let ptr = vec.clone().into_raw_parts().0.into();
        forget(vec);
        VecArr {
            ptr,
            capacity: self.vec_capacity().into(),
        }
    }
}

pub struct VecIter<'a, T> {
    ptr: *mut T,
    start: usize,
    end: usize,
    _p: PhantomData<&'a mut T>,
}
impl<'a, T> VecIter<'a, T> {
    #[inline(always)]
    pub fn empty() -> Self {
        VecIter {
            ptr: null_mut(),
            start: 0,
            end: 0,
            _p: PhantomData,
        }
    }
    #[inline(always)]
    fn new(ptr: *mut T, start: usize, end: usize) -> Self {
        VecIter {
            ptr,
            start,
            end,
            _p: PhantomData,
        }
    }

    #[inline(always)]
    pub fn index(&self) -> usize {
        self.start
    }
}
impl<'a, T> Iterator for VecIter<'a, T> {
    type Item = &'a mut T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let r = unsafe { transmute(self.ptr.add(self.start)) };
            self.start += 1;
            return Some(r);
        }
        return None;
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let min = self.end.saturating_sub(self.start);
        return (min, Some(min));
    }
}

/// take vec.
pub fn to_vec<T>(ptr: *mut T, len: usize) -> Vec<T> {
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

#[cfg(test)]
mod tests {
    use pcg_rand::Pcg64;
    use pi_buckets::{Location, MAX_ENTRIES};
    use rand::{Rng, SeedableRng};
    use std::sync::{Arc, Mutex};

    use crate::*;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct A();

    #[test]
    fn test1() {
        let arr: Arr<u8> = arr![1; 3];
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 1);
    }
    #[test]
    fn test22() {
        println!("test22 start");
        //let mut rng = rand::thread_rng();
        let mut rng = Pcg64::seed_from_u64(300000);
        let mut arr = arr![];
        let mut vec = vec![];
        for _c in 0..1000 {
            // println!("test22 start c:{}", c);
            arr.clear(vec.len(), 0);
            vec.clear();
            let x = rng.gen_range(0..100) + 2;
            for _ in 0..x {
                let r = rng.gen_range(1..10000);
                // println!("r: {:?}", r);
                arr.set(r, r);
                if vec.len() <= r {
                    vec.resize(r + 1, 0);
                }
                vec[r] = r;
            }
            match_arr_vec(&vec, &arr);
            let j = rng.gen_range(vec.len() / 2..vec.len());
            let k = rng.gen_range(0..j);
            // println!("vec: {:?}", vec.iter().filter(|r| **r > 0).collect::<Vec<_>>());
            // println!("kj: {:?}, len:{}", (k, j), vec.len());
            arr.remain_settle(k..j, vec.len(), 0);
            vec.remain(k..j);
            match_arr_vec(&vec, &arr);
        }
    }
    fn match_arr_vec(vec: &Vec<usize>, arr: &Arr<usize>) {
        // println!(
        //     "match_arr_vec vec: {:?}, len:{:?}",
        //     vec.iter().enumerate().filter(|r| *r.1 > 0).collect::<Vec<_>>(),
        //     vec.len()
        // );
        // println!(
        //     "match_arr_vec arr: {:?}",
        //     arr.slice(0..vec.len()).enumerate().filter(|r| *r.1 > 0).collect::<Vec<_>>(),
        // );
        for i in 0..vec.len() {
            if vec[i] == 0 {
                continue;
            }
            assert_eq!(vec[i], arr[i], "i: {}", i);
        }
    }
    #[test]
    fn test2() {
        println!("test2 start");
        let mut arr = arr![];
        let mut i = 0;
        //let mut rng = rand::thread_rng();
        let mut rng = Pcg64::seed_from_u64(1);
        for _ in 0..1000 {
            let x = rng.gen_range(0..1000);
            for _ in 0..x {
                arr.insert(i, i);
                i += 1;
            }
            check(&arr, i);
            if rng.gen_range(0..200) == 0 {
                arr.clear(i, 0);
                i = 0;
            }
            if rng.gen_range(0..100) == 0 && i > 20 {
                let j = rng.gen_range(0..20);
                arr.remain_settle(j..i, i, rng.gen_range(0..100));
                for k in 0..i - j {
                    assert_eq!(arr[k], k + j);
                }
                arr.clear(i - j, 0);
                i = 0;
            }
            arr.settle(i, rng.gen_range(0..100));
            if rng.gen_range(0..200) == 0 {
                arr.clear(i, 0);
                i = 0;
            }
            check1(&arr, i);
        }
        println!("test2 arr.vec_capacity(): {}", arr.vec_capacity());
    }
    fn check(arr: &Arr<usize>, len: usize) {
        for i in 0..len {
            assert_eq!(arr[i], i);
        }
    }
    fn check1(arr: &Arr<usize>, len: usize) {
        for i in arr.slice(0..len).enumerate() {
            assert_eq!(i.0, *i.1);
        }
    }

    #[cfg(not(feature = "rc"))]
    #[test]
    fn test() {
        let arr = arr![1; 3];
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 1);
        assert_eq!(arr[2], 1);

        let mut arr = arr![10, 40, 30];
        assert_eq!(40, *arr.alloc(1));
        assert_eq!(0, *arr.alloc(3));
        assert_eq!(40, arr.set(1, 20));
        assert_eq!(0, arr.set(33, 33));

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
        assert_eq!(0, *arr.load_alloc(3));
        assert_eq!(0, *arr.load_alloc(133));

        let arr = arr![10, 40, 30];
        assert_eq!(40, *arr.load(1).unwrap());
        assert_eq!(None, arr.load(3));
        assert_eq!(None, arr.load(133));

        let arr = crate::arr![1, 2];
        arr.insert(2, 3);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);

        let arr = crate::arr![1, 2, 4];
        arr.insert(97, 97);
        println!("arr: {:?}", arr.vec_capacity());
        let mut iterator = arr.slice(0..160).enumerate();
        println!("arr: {:?}", iterator.size_hint());
        // assert_eq!(iterator.size_hint().0, 3);
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (0, 1));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (1, 2));
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (2, 4));
        for i in 3..32 {
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1), (i, 0));
        }
        for i in 32..65 {
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1), (i, 0));
        }
        let r = iterator.next().unwrap();
        assert_eq!((r.0, *r.1), (65, 97));
        for i in 66..67 {
            let r = iterator.next().unwrap();
            assert_eq!((r.0, *r.1), (i, 0));
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
    fn test_zst() {
        let mut vec = Vec::new();
        println!("test_zst vec: {}", vec.capacity());
        vec.insert(0, A());
        vec.insert(1, A());
        let a = vec.get(0);
        println!("test_zst vec a: {:?}", (a, Some(&A())));
        let mut arr = Arr::new();
        println!("test_zst: {}", arr.vec_capacity());
        arr.set(0, A());
        arr.set(1, A());
        let a = arr.get(0);
        println!("test_zst a: {:?}", (a, Some(&A())));
        assert_eq!(a, Some(&A()));
        assert_eq!(arr.get(1), Some(&A()));
        assert_eq!(arr.get(2), Some(&A()));
        let arr = arr.clone();
        let a: Option<&A> = arr.get(0);
        println!("test_zst a: {:?}", (a, Some(&A())));
        assert_eq!(a, Some(&A()));
        assert_eq!(arr.get(1), Some(&A()));
        assert_eq!(arr.get(2), Some(&A()));
        let it: BucketIter<'_, A> = arr.slice(0..1000);
        println!("test_zst it: {:?}", it.size_hint());
        for i in it {
            assert_eq!(i, &A());
        }
    }
    #[test]
    fn test_arc() {
        let arr = Arc::new(crate::Buckets::new());

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
    fn test_arc2() {
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
            assert!(arr.slice(0..MAX_ENTRIES).any(|x| *x == i));
        }
    }
    #[test]
    fn test_arc11() {
        let vec: Vec<Option<Arc<usize>>> = Vec::new();
        let vec1 = vec.clone();
        println!(
            "test_arc1 start:, {:p} {:?}",
            vec1.into_raw_parts().0,
            std::ptr::null::<usize>()
        );
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
        println!("test_arc1 {:?}", a1[0]);
    }
    #[test]
    fn test_arc1() {
        let vec: Vec<Option<Arc<usize>>> = Vec::new();
        let vec1 = vec.clone();
        println!("test_arc1 start:, {:?}", vec1.into_raw_parts().0.is_null());
        let a1 = {
            let arr = crate::Buckets::new();
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
        println!("test_arc1 {:?}", a1[0]);
    }
    #[test]
    fn test_mutex() {
        let arr = Arc::new(crate::Buckets::new());

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
}
