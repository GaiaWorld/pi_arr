//! 自动扩展的数组VecArr，由一个可自动扩容的Vec实现。
//! 自动扩展的数组VBArr，由一个可扩展槽加多个固定槽构成，每个固定槽用不扩容的Vec来装元素。
//! 当槽位上的Vec长度不够时，不会立刻扩容Vec，而是线程安全的到下一个槽位分配新Vec。
//! 第一个固定槽位的Vec长度为32。
//! 固定槽迭代性能比Vec慢1-10倍， 主要损失在切换bucket时，原子操作及缓存失效。
//! 在整理时，会一次性将所有固定槽元素移动到扩展槽。

#![feature(unsafe_cell_access)]
#![feature(vec_into_raw_parts)]
#![feature(test)]
extern crate test;

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::{forget, replace, size_of, transmute};
use std::ops::{Index, IndexMut, Range};
use std::ptr::{null, null_mut, NonNull};
use std::sync::atomic::Ordering;

use pi_share::{ShareMutex, SharePtr};
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
    buckets: *mut BucketArr<T>,
}
#[cfg(not(feature = "rc"))]
impl<T> Default for VBArr<T> {
    fn default() -> Self {
        let buckets = if size_of::<T>() == 0 {
            NonNull::<BucketArr<T>>::dangling().as_ptr()
        } else {
            Box::into_raw(Box::new(BucketArr::default()))
        };
        Self {
            ptr: NonNull::<T>::dangling().as_ptr(),
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
        Self::with_capacity_multiple(capacity, 1)
    }

    pub fn with_capacity_multiple(capacity: usize, multiple: usize) -> VBArr<T> {
        if size_of::<T>() == 0 || capacity == 0 {
            return VBArr::default();
        }
        let buckets = Box::into_raw(Box::new(Default::default()));
        return VBArr {
            ptr: bucket_alloc(capacity * multiple),
            capacity,
            buckets,
        };
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
    pub unsafe fn set_vec_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
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
    fn buckets(&self) -> &BucketArr<T> {
        unsafe { &*self.buckets }
    }
    #[inline(always)]
    fn buckets_mut(&mut self) -> &mut BucketArr<T> {
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
    /// it will be allocated automatically, and the returned T is null
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
    /// it will be allocated automatically, and the returned T is null
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
    /// let mut iterator = arr.slice(0..pi_arr::MAX_ENTRIES);
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
                &self.buckets().buckets,
                self.capacity,
            )
        } else if range.start < self.capacity {
            let end = Location::of(range.end - self.capacity);
            BucketIter::new(
                self.ptr,
                Location::new(-1, self.capacity, range.start),
                end.entry,
                end.bucket,
                &self.buckets().buckets,
                self.capacity,
            )
        } else {
            self.buckets().slice_row(range, self.capacity)
        }
    }
    #[inline]
    pub fn get_multiple(&self, index: usize, multiple: usize) -> Option<&mut T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &mut *self.ptr.add(index * multiple) });
        }
        let mut loc = Location::of(index - self.capacity);
        loc.entry *= multiple;
        self.buckets().load(&loc)
    }

    #[inline]
    pub fn get_multiple_unchecked(&self, index: usize, multiple: usize) -> &mut T {
        if index < self.vec_capacity() {
            return unsafe { &mut *self.ptr.add(index * multiple) };
        }
        let mut loc = Location::of(index - self.capacity);
        loc.entry *= multiple;
        unsafe { self.buckets().load_unchecked(&loc) }
    }

    #[inline]
    pub fn load_alloc_multiple(&self, index: usize, multiple: usize) -> &mut T {
        if index < self.vec_capacity() {
            return unsafe { &mut *self.ptr.add(index * multiple) };
        }
        let mut loc = Location::of(index - self.capacity);
        loc.entry *= multiple;
        loc.len *= multiple;
        self.buckets().load_alloc(&loc)
    }
    /// 保留范围内的数组元素并整理，将保留的部分整理到扩展槽中，并将当前vec_capacity容量扩容len+additional
    pub fn remain_settle(
        &mut self,
        range: Range<usize>,
        len: usize,
        additional: usize,
        multiple: usize,
    ) {
        if size_of::<T>() == 0 || multiple == 0 {
            return;
        }
        debug_assert!(len >= range.end);
        debug_assert!(range.end >= range.start);
        let mut vec = to_vec(self.ptr, self.capacity * multiple);
        if range.end <= self.capacity {
            // 数据都在vec上
            vec.remain(range.start * multiple..range.end * multiple);
            if len > self.capacity {
                self.take_buckets(multiple);
            }
            return self.reserve(vec, range.len(), additional, multiple);
        }
        // 取出所有的bucket
        let arr = self.take_buckets(multiple);
        // 获得扩容后的总容量
        let cap = Location::bucket_capacity(Location::bucket(range.len() + additional)) * multiple;
        let mut start = range.start * multiple;
        let end = range.end * multiple;
        let mut index = vec.capacity();
        if vec.capacity() >= cap {
            // 先将扩展槽的数据根据范围保留
            start += vec.remain(start..end);
        } else if vec.capacity() > 0 {
            let mut new = Vec::with_capacity(cap);
            // 将扩展槽的数据根据范围保留到新vec中
            start += vec.remain_to(start..end, &mut new);
            vec = new;
        }
        // 将arr的数据移到vec上
        for (i, mut v) in arr.into_iter().enumerate() {
            if start >= end {
                break;
            }
            let mut vlen = v.len();
            if vlen > 0 {
                if start >= index + vlen {
                    index += vlen;
                    continue;
                }
                if vec.capacity() == 0 {
                    // 如果原vec为empty
                    if v.capacity() >= cap {
                        // 并且当前容量大于等于cap，则直接将v换上
                        _ = replace(&mut vec, v);
                        vec.remain(start - index..end - index);
                    } else {
                        vec.reserve(cap);
                        v.remain_to(start - index..end - index, &mut vec);
                    }
                } else {
                    v.remain_to(start - index..end - index, &mut vec);
                }
            } else {
                vlen = Location::bucket_len(i) * multiple;
                if start >= index + vlen {
                    index += vlen;
                    continue;
                }
                if vec.capacity() == 0 {
                    vec.reserve(cap);
                }
                vec.resize_with(vec.len() + index + vlen - start, || T::default());
            }
            index += vlen;
            start = index;
        }
        // 如果容量比len大，则初始化为null元素
        vec.resize_with(vec.capacity(), || T::default());
        self.capacity = vec.capacity() / multiple;
        self.ptr = vec.into_raw_parts().0;
    }
    /// 整理内存，将bucket_arr的数据移到vec上，并将当前vec_capacity容量扩容len+additional
    pub fn settle(&mut self, len: usize, additional: usize, multiple: usize) {
        self.remain_settle(0..len, len, additional, multiple);
    }
    /// 清理所有数据，释放bucket_arr的内存，并尝试将当前vec_capacity容量扩容len+additional
    /// 释放前，必须先调用clear方法，保证释放其中的数据
    #[inline(always)]
    pub fn clear(&mut self, len: usize, additional: usize, multiple: usize) {
        if size_of::<T>() == 0 || multiple == 0 {
            return;
        }
        if len > self.capacity {
            Self::reset_vec(self.buckets().take(), multiple);
        }
        let mut vec = to_vec(self.ptr, self.capacity * multiple);
        vec.clear();
        self.reserve(vec, len, additional, multiple);
    }
    fn reserve(&mut self, mut vec: Vec<T>, len: usize, mut additional: usize, multiple: usize) {
        additional = (len + additional).saturating_sub(self.capacity);
        if additional > 0 {
            vec.reserve(additional * multiple);
            vec.resize_with(vec.capacity(), || T::default());
            self.capacity = vec.capacity() / multiple;
        }
        self.ptr = vec.into_raw_parts().0;
    }
    fn take_buckets(&mut self, multiple: usize) -> [Vec<T>; BUCKETS] {
        // 取出所有的bucket
        let mut arr = self.buckets().take();
        if multiple > 1 {
            arr = Self::reset_vec(arr, multiple);
        }
        arr
    }
    fn reset_vec(buckets: [Vec<T>; BUCKETS], multiple: usize) -> [Vec<T>; BUCKETS] {
        buckets.map(|vec| {
            let len = vec.len() * multiple;
            let ptr = vec.into_raw_parts().0;
            to_vec(ptr, len)
        })
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
        Self::with_capacity_multiple(capacity, 1)
    }
    pub fn with_capacity_multiple(capacity: usize, multiple: usize) -> VecArr<T> {
        if multiple == 0 {
            return Self {
                ptr: NonNull::<T>::dangling().as_ptr().into(),
                capacity: usize::MAX.into(),
            }
        }
        if size_of::<T>() == 0 || capacity == 0 {
            return VecArr::default();
        }
        return VecArr {
            ptr: bucket_alloc::<T>(capacity * multiple).into(),
            capacity: capacity.into(),
        };
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
    /// it will be allocated automatically, and the returned T is null
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
            self.reserve(vec, self.vec_capacity(), index - self.vec_capacity() + 1, 1);
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
    /// it will be allocated automatically, and the returned T is null
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
            self.reserve(vec, self.vec_capacity(), index - self.vec_capacity() + 1, 1);
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
    /// let mut iterator = arr.slice(0..pi_arr::MAX_ENTRIES);
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
    #[inline]
    pub fn get_multiple(&self, index: usize, multiple: usize) -> Option<&mut T> {
        if index < self.vec_capacity() {
            return Some(unsafe { &mut *(*self.ptr.get()).add(index * multiple) });
        }
        None
    }
    #[inline]
    pub fn get_multiple_unchecked(&self, index: usize, multiple: usize) -> &mut T {
        debug_assert!(index < self.vec_capacity());
        unsafe { &mut *(*self.ptr.get()).add(index * multiple) }
    }
    #[inline]
    pub fn load_alloc_multiple(&self, index: usize, multiple: usize) -> &mut T {
        if index >= self.vec_capacity() {
            let vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity() * multiple);
            self.reserve(
                vec,
                self.vec_capacity(),
                index - self.vec_capacity() + 1,
                multiple,
            );
        }
        return unsafe { &mut *(*self.ptr.get()).add(index * multiple) };
    }
    /// 保留范围内的数组元素并整理，将保留的部分整理到扩展槽中，并将当前vec_capacity容量扩容len+additional
    pub fn remain_settle(
        &mut self,
        range: Range<usize>,
        _len: usize,
        additional: usize,
        multiple: usize,
    ) {
        if size_of::<T>() == 0 || multiple == 0 {
            return;
        }
        let mut vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity() * multiple);
        // 数据都在vec上
        vec.remain(range.start * multiple..range.end * multiple);
        return self.reserve(vec, range.len(), additional, multiple);
    }
    /// 整理内存，将bucket_arr的数据移到vec上，并将当前vec_capacity容量扩容len+additional
    pub fn settle(&mut self, _len: usize, _additional: usize, _multiple: usize) {}
    /// 清理所有数据，释放bucket_arr的内存，并尝试将当前vec_capacity容量扩容len+additional
    /// 释放前，必须先调用clear方法，保证释放其中的数据
    #[inline(always)]
    pub fn clear(&mut self, len: usize, additional: usize, multiple: usize) {
        if size_of::<T>() == 0 || multiple == 0 {
            return;
        }
        let mut vec = to_vec(unsafe { *self.ptr.get() }, self.vec_capacity() * multiple);
        vec.clear();
        self.reserve(vec, len, additional, multiple);
    }
    fn reserve(&self, mut vec: Vec<T>, len: usize, mut additional: usize, multiple: usize) {
        additional = (len + additional).saturating_sub(self.vec_capacity());
        if additional > 0 {
            vec.reserve(additional * multiple);
            vec.resize_with(vec.capacity(), || T::default());
            unsafe { self.capacity.replace(vec.capacity() / multiple) };
        }
        unsafe { self.ptr.replace(vec.into_raw_parts().0) };
    }
}
impl<T> Drop for VecArr<T> {
    fn drop(&mut self) {
        if size_of::<T>() == 0 {
            return;
        }
        let len = unsafe {
            *self.capacity.as_ref_unchecked()
        };
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

pub const BUCKETS: usize = (u32::BITS as usize) - SKIP_BUCKET;
pub const MAX_ENTRIES: usize = (u32::MAX as usize) - SKIP;

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
macro_rules! barr {
    () => {
        $crate::Arr::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut arr = $crate::BucketArr::with_capacity($n);
        arr.extend(::core::iter::repeat($elem).take($n));
        arr
    }};
    ($($x:expr),+ $(,)?) => (
        <$crate::BucketArr<_> as core::iter::FromIterator<_>>::from_iter([$($x),+])
    );
}

/// A lock-free, auto-expansion array by buckets.
///
/// See [the crate documentation](crate) for details.
///
/// # Notes
///
/// The bucket array is stored inline, meaning that the
/// `Arr<T>` is quite large. It is expected that you
/// store it behind an [`Arc`](std::sync::Arc) or similar.
pub struct BucketArr<T> {
    buckets: [SharePtr<T>; BUCKETS],
    lock: ShareMutex<()>,
}
#[cfg(not(feature = "rc"))]
impl<T> Default for BucketArr<T> {
    fn default() -> Self {
        let buckets = [null_mut(); BUCKETS];
        BucketArr {
            buckets: buckets.map(SharePtr::new),
            lock: ShareMutex::default(),
        }
    }
}

unsafe impl<T: Send> Send for BucketArr<T> {}
unsafe impl<T: Sync> Sync for BucketArr<T> {}

#[cfg(not(feature = "rc"))]
impl<T: Default> BucketArr<T> {
    /// Constructs a new, empty `Arr<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::pi_arr;
    /// let arr: pi_arr::Arr<i32> = pi_arr::Arr::new();
    /// ```
    #[inline]
    pub fn new() -> BucketArr<T> {
        BucketArr::default()
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
    /// use crate::{pi_arr, pi_arr::Location};
    /// let mut arr = pi_arr::BucketArr::with_capacity(10);
    ///
    /// for i in 0..32 {
    ///     // will not allocate
    ///     arr.set(&Location::of(i), i);
    /// }
    ///
    /// // may allocate
    /// arr.set(&Location::of(33), 33);
    /// ```
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> BucketArr<T> {
        Self::with_capacity_multiple(capacity, 1)
    }

    pub fn with_capacity_multiple(capacity: usize, multiple: usize) -> BucketArr<T> {
        let mut buckets = [null_mut(); BUCKETS];
        if capacity == 0 {
            return BucketArr {
                buckets: buckets.map(SharePtr::new),
                lock: ShareMutex::default(),
            };
        }
        let end = Location::of(capacity).bucket as usize;
        for (i, bucket) in buckets[..=end].iter_mut().enumerate() {
            let len = Location::bucket_len(i);
            *bucket = bucket_alloc(len * multiple);
        }

        BucketArr {
            buckets: buckets.map(SharePtr::new),
            lock: ShareMutex::default(),
        }
    }

    /// Returns a reference to the element at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// use crate::{pi_arr, pi_arr::Location};
    /// let arr = pi_arr::barr![10, 40, 30];
    /// assert_eq!(Some(&40), arr.get(&Location::of(1)));
    /// assert_eq!(None, arr.get(&Location::of(33)));
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
    /// use crate::{pi_arr, pi_arr::Location};
    /// let arr = pi_arr::barr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.get_unchecked(&Location::of(1)), &2);
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
    /// use crate::{pi_arr, pi_arr::Location};
    /// let mut arr = pi_arr::barr![10, 40, 30];
    /// assert_eq!(Some(&mut 40), arr.get_mut(&Location::of(1)));
    /// assert_eq!(None, arr.get_mut(&Location::of(33)));
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
    /// use crate::{pi_arr, pi_arr::Location};
    /// let mut arr = pi_arr::barr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.get_unchecked_mut(&Location::of(1)), &mut 2);
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
    ///
    /// use crate::{pi_arr, pi_arr::Location};
    /// let mut arr = pi_arr::barr![10, 40, 30];
    /// assert_eq!(40, *arr.alloc(&Location::of(1)));
    /// assert_eq!(0, *arr.alloc(&Location::of(3)));
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
    ///
    /// use crate::{pi_arr, pi_arr::Location};
    /// let mut arr = crate::pi_arr::barr![10, 40, 30];
    /// assert_eq!(40, arr.set(&Location::of(1), 20));
    /// assert_eq!(Some(&20), arr.get(&Location::of(1)));
    /// assert_eq!(0, arr.set(&Location::of(33), 5));
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
    ///
    /// use crate::{pi_arr, pi_arr::Location};
    /// let arr = pi_arr::barr![10, 40, 30];
    /// assert_eq!(10, *arr.load(&Location::of(0)).unwrap());
    /// assert_eq!(Some(&mut 40), arr.load(&Location::of(1)));
    /// assert_eq!(0, *arr.load(&Location::of(3)).unwrap());
    /// assert_eq!(None, arr.load(&Location::of(33)));
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
    /// use crate::{pi_arr, pi_arr::Location};
    /// let arr = pi_arr::barr![1, 2, 4];
    ///
    /// unsafe {
    ///     assert_eq!(arr.load_unchecked(&Location::of(1)), &mut 2);
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
    ///
    /// use crate::{pi_arr, pi_arr::Location};
    /// let arr = pi_arr::barr![10, 40, 30];
    /// assert_eq!(40, *arr.load_alloc(&Location::of(1)));
    /// assert_eq!(0, *arr.load_alloc(&Location::of(3)));
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
    /// use crate::{pi_arr, pi_arr::Location};
    /// let arr = pi_arr::barr![1, 2];
    /// arr.insert(&Location::of(2), 3);
    /// assert_eq!(arr[0], 1);
    /// assert_eq!(arr[1], 2);
    /// assert_eq!(arr[2], 3);
    /// ```
    #[inline(always)]
    pub fn insert(&self, location: &Location, value: T) -> T {
        replace(self.load_alloc(location), value)
    }
    /// take buckets.
    pub fn take(&self) -> [Vec<T>; BUCKETS] {
        let mut buckets = [0; BUCKETS].map(|_| Vec::new());
        for (i, p) in self.buckets.iter().enumerate() {
            let ptr = p.swap(null_mut(), Ordering::Relaxed);
            if ptr.is_null() {
                continue;
            }
            buckets[i] = to_bucket_vec(ptr, i);
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
    ///
    /// use crate::{pi_arr, pi_arr::Location};
    /// let arr = pi_arr::barr![1, 2, 4];
    /// arr.insert(&Location::of(98), 98);
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
    ///     assert_eq!((iterator.index() - 1, *r), (i, 0));
    /// }
    /// for i in 96..98 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1, *r), (i, 0));
    /// }
    /// let r = iterator.next().unwrap();
    /// assert_eq!((iterator.index() - 1, *r), (98, 98));
    /// for i in 99..224 {
    ///     let r = iterator.next().unwrap();
    ///     assert_eq!((iterator.index() - 1, *r), (i, 0));
    /// }
    /// assert_eq!(iterator.next(), None);
    /// assert_eq!(iterator.size_hint().0, 0);
    /// ```
    #[inline]
    pub fn iter(&self) -> BucketIter<'_, T> {
        self.slice_row(0..MAX_ENTRIES, 0)
    }

    /// Returns an iterator over the array at the given range.
    ///
    /// Values are yielded in the form `Entry`.
    ///
    /// # Examples
    ///
    /// ```
    /// let arr = crate::pi_arr::barr![1, 2, 4, 6];
    /// let mut iterator = arr.slice(1..3);
    ///
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 2);
    /// let r = iterator.next().unwrap();
    /// assert_eq!(*r, 4);
    /// assert_eq!(iterator.next(), None);
    /// ```
    pub fn slice(&self, range: Range<usize>) -> BucketIter<'_, T> {
        self.slice_row(range, 0)
    }
    fn slice_row(&self, range: Range<usize>, capacity: usize) -> BucketIter<'_, T> {
        let start = Location::of(range.start - capacity);
        let end = Location::of(range.end - capacity);
        BucketIter::new(
            null_mut(),
            start,
            end.entry,
            end.bucket,
            &self.buckets,
            capacity,
        )
        .init_iter()
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

#[cfg(not(feature = "rc"))]
impl<T: Default> Index<usize> for BucketArr<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        self.get(&Location::of(index))
            .expect("no element found at index {index}")
    }
}
#[cfg(not(feature = "rc"))]
impl<T: Default> IndexMut<usize> for BucketArr<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(&Location::of(index))
            .expect("no element found at index_mut {index}")
    }
}
#[cfg(not(feature = "rc"))]
impl<T> Drop for BucketArr<T> {
    fn drop(&mut self) {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            let ptr = *bucket.get_mut();
            if ptr.is_null() {
                continue;
            }
            // safety: in drop
            to_bucket_vec(ptr, i);
        }
    }
}


#[cfg(not(feature = "rc"))]
impl<T: Default> FromIterator<T> for BucketArr<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();

        let (lower, _) = iter.size_hint();
        let mut arr = BucketArr::with_capacity(lower);
        for (i, value) in iter.enumerate() {
            arr.set(&Location::of(i), value);
        }
        arr
    }
}

#[cfg(not(feature = "rc"))]
impl<T: Default> Extend<T> for BucketArr<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        for (i, value) in iter.enumerate() {
            self.set(&Location::of(i), value);
        }
    }
}

#[cfg(not(feature = "rc"))]
impl<T: Default + Clone> Clone for BucketArr<T> {
    fn clone(&self) -> BucketArr<T> {
        let mut buckets: [*mut T; BUCKETS] = [null_mut(); BUCKETS];

        for (i, bucket) in buckets.iter_mut().enumerate() {
            let ptr = unsafe { self.load_entries(i) };
            if ptr.is_null() {
                continue;
            }
            let vec = to_bucket_vec(ptr, i);
            *bucket = vec.clone().into_raw_parts().0;
            forget(vec);
        }
        BucketArr {
            buckets: buckets.map(SharePtr::new),
            lock: ShareMutex::default(),
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

/// An iterator over the elements of a [`Arr<T>`].
///
/// See [`Arr::iter`] for details.
pub struct BucketIter<'a, T> {
    ptr: *mut T,
    start: Location,
    end_entry: usize,
    end_bucket: isize,
    buckets: &'a [SharePtr<T>; BUCKETS],
    capacity: usize,
}

#[cfg(not(feature = "rc"))]
impl<'a, T> BucketIter<'a, T> {
    #[inline(always)]
    pub fn empty() -> Self {
        BucketIter {
            ptr: null_mut(),
            start: Location::default(),
            end_entry: 0,
            end_bucket: 0,
            buckets: unsafe { transmute(null::<[SharePtr<T>; BUCKETS]>()) },
            capacity: 0,
        }
    }
    #[inline(always)]
    fn new(
        ptr: *mut T,
        start: Location,
        end_entry: usize,
        end_bucket: isize,
        buckets: &'a [SharePtr<T>; BUCKETS],
        capacity: usize,
    ) -> Self {
        BucketIter {
            ptr,
            start,
            end_entry,
            end_bucket,
            buckets,
            capacity,
        }
    }
    #[inline(always)]
    fn init_iter(mut self) -> Self {
        if self.start.bucket > self.end_bucket {
            self.start.len = self.start.entry;
            return self;
        }
        if self.start.bucket == self.end_bucket {
            self.start.len = self.end_entry;
            if self.start.len == 0 {
                return self;
            }
        }
        self.load_ptr();
        if self.ptr.is_null() {
            self.start.len = self.start.entry;
        }
        self
    }
    #[inline(always)]
    pub fn index(&self) -> usize {
        self.start.index(self.capacity)
    }
    #[inline(always)]
    pub(crate) fn get(&mut self) -> &'a mut T {
        unsafe { transmute(self.ptr.add(self.start.entry)) }
    }
    #[inline(always)]
    fn load_ptr(&mut self) {
        self.ptr = unsafe {
            self.buckets
                .get_unchecked(self.start.bucket as usize)
                .load(Ordering::Relaxed)
        };
    }
    #[inline]
    pub(crate) fn next_bucket(&mut self) -> Option<&'a mut T> {
        loop {
            if self.start.bucket >= self.end_bucket {
                return None;
            }
            self.start.bucket += 1;
            self.load_ptr();
            if self.ptr.is_null() {
                continue;
            }
            if self.start.bucket == self.end_bucket {
                if self.end_entry == 0 {
                    return None;
                }
                self.start.len = self.end_entry;
            } else {
                self.start.len = Location::bucket_len(self.start.bucket as usize);
            }
            self.start.entry = 1;
            return Some(unsafe { transmute(self.ptr) });
        }
    }
    fn size(&self) -> (usize, Option<usize>) {
        if self.start.bucket > self.end_bucket {
            return (0, Some(0));
        }
        // 最小为起始槽的entry数量
        let min = self.start.len.saturating_sub(self.start.entry);
        // println!("size: {:?}", (min, self.start.len, self.start.entry));
        let c = self.end_bucket - self.start.bucket;
        if c == 0 {
            return (min, Some(min));
        }
        if c == 1 {
            return (min, Some(min + self.end_entry));
        }
        if self.start.bucket < 0 {
            let end = Location::new(c - 1, 0, self.end_entry);
            return (min, Some(min + end.index(0)));
        }
        // 中间槽的entry数量
        let n = self.start.len * (1 << (c - 1));
        (min, Some(min + n + self.end_entry))
    }
}
#[cfg(not(feature = "rc"))]
impl<'a, T> Iterator for BucketIter<'a, T> {
    type Item = &'a mut T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.start.entry < self.start.len {
                let r = self.get();
                self.start.entry += 1;
                return Some(r);
            }
            // return None
            return self.next_bucket();
            // 将代码内联后， bench_arr的性能由10ns变为40ns
            // loop {
            //     if self.start.bucket >= self.end_bucket {
            //         return None;
            //     }
            //     self.start.bucket += 1;
            //     self.load_ptr();
            //     if !self.ptr.is_null() {
            //         break;
            //     }
            // }
            // self.start.entry = 0;
            // if self.start.bucket == self.end_bucket {
            //     self.start.len = self.end_entry;
            // } else {
            //     self.start.len = Location::bucket_len(self.start.bucket as usize);
            // }
        }
    }
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.size()
    }
}
/// take vec.
pub fn to_vec<T>(ptr: *mut T, len: usize) -> Vec<T> {
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

/// take vec.
pub fn to_bucket_vec<T>(ptr: *mut T, bucket: usize) -> Vec<T> {
    let len = Location::bucket_len(bucket);
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

fn bucket_alloc<T: Default>(len: usize) -> *mut T {
    let mut entries: Vec<T> = Vec::with_capacity(len);
    entries.resize_with(entries.capacity(), || T::default());
    entries.into_raw_parts().0
}
#[cfg(not(feature = "rc"))]
fn bucket_init<T: Default>(share_ptr: &SharePtr<T>, len: usize, lock: &ShareMutex<()>) -> *mut T {
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
    // the length
    pub len: usize,
    // the index of the entry in `bucket`
    pub entry: usize,
    // the index of the bucket
    pub bucket: isize,
}

impl Location {
    #[inline(always)]
    pub const fn new(bucket: isize, bucket_len: usize, entry: usize) -> Self {
        Location {
            len: bucket_len,
            entry,
            bucket,
        }
    }
    #[inline(always)]
    pub const fn bucket(index: usize) -> usize {
        let skipped = index.checked_add(SKIP).expect("exceeded maximum length");
        let bucket = usize::BITS - skipped.leading_zeros();
        (bucket as usize) - (SKIP_BUCKET + 1)
    }
    #[inline(always)]
    pub const fn of(index: usize) -> Location {
        let skipped = index.checked_add(SKIP).expect("exceeded maximum length");
        let bucket = usize::BITS - skipped.leading_zeros();
        let bucket = (bucket as usize) - (SKIP_BUCKET + 1);
        let bucket_len = Location::bucket_len(bucket);
        let entry = skipped ^ bucket_len;

        Location {
            len: bucket_len,
            entry,
            bucket: bucket as isize,
        }
    }
    #[inline(always)]
    pub const fn bucket_len(bucket: usize) -> usize {
        1 << (bucket + SKIP_BUCKET)
    }
    #[inline(always)]
    pub const fn bucket_capacity(bucket: usize) -> usize {
        (1 << (bucket + SKIP_BUCKET + 1)) - SKIP
    }
    #[inline(always)]
    pub const fn index(&self, capacity: usize) -> usize {
        if self.bucket < 0 {
            return self.entry;
        }
        ((i32::MAX as u32) >> (u32::BITS - 1 - self.bucket as u32) << SKIP_BUCKET) as usize
            + self.entry
            + capacity
    }
}

#[cfg(test)]
mod tests {
    use pcg_rand::Pcg64;
    use rand::{Rng, SeedableRng};
    use std::sync::{Arc, Mutex};

    use test::Bencher;

    use crate::*;
    static mut AAA: u64 = 0;

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
        let mut rng = rand::thread_rng();
        //let mut rng = Pcg64::seed_from_u64(1);
        for _c in 0..1000 {
            let mut arr = arr![];
            let mut vec = vec![];
            // println!("test22 start c:{}", c);
            arr.clear(vec.len(), 0, 1);
            vec.clear();
            let x = rng.gen_range(0..100) + 2;
            for _ in 0..x {
                let r = rng.gen_range(1..1000);
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
            //println!("kj: {:?}, len:{}", (k, j), vec.len());
            arr.remain_settle(k..j, vec.len(), 0, 1);
            vec.remain(k..j);
            match_arr_vec(&vec, &arr);
        }
    }
    fn match_arr_vec(vec: &Vec<usize>, arr: &Arr<usize>) {
        // println!(
        //     "match_arr_vec vec: {:?}, len:{:?}",
        //     vec.iter().filter(|r| **r > 0).collect::<Vec<_>>(),
        //     vec.len()
        // );
        for i in 0..vec.len() {
            if vec[i] == 0 {
                continue;
            }
            assert_eq!(vec[i], arr[i]);
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
                arr.clear(i, 0, 1);
                i = 0;
            }
            if rng.gen_range(0..100) == 0 && i > 20 {
                let j = rng.gen_range(0..20);
                arr.remain_settle(j..i, i, rng.gen_range(0..100), 1);
                for k in 0..i - j {
                    assert_eq!(arr[k], k + j);
                }
                arr.clear(i - j, 0, 1);
                i = 0;
            }
            arr.settle(i, rng.gen_range(0..100), 1);
            if rng.gen_range(0..200) == 0 {
                arr.clear(i, 0, 1);
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
    #[test]
    fn test3() {
        println!("test3 start");
        let mut arr = Arr::<u8>::new();
        let mut i = 0;
        // let mut rng = rand::thread_rng();
        let mut rng = Pcg64::seed_from_u64(1);
        for _ in 0..1000 {
            let x = rng.gen_range(0..1000);
            for _ in 0..x {
                let r: &mut usize =
                    unsafe { transmute(arr.load_alloc_multiple(i, size_of::<usize>())) };
                *r = i;
                i += 1;
            }
            check3(&arr, i);
            if rng.gen_range(0..200) == 0 {
                arr.clear(i, 0, size_of::<usize>());
                i = 0;
            }
            if rng.gen_range(0..100) == 0 && i > 20 {
                let j = rng.gen_range(0..20);
                println!("test3: c:{:?}, ij: {:?}", arr.vec_capacity(), (i, j));
                arr.remain_settle(j..i, i, rng.gen_range(0..100), size_of::<usize>());
                for k in 0..i - j {
                    let r: &mut usize =
                        unsafe { transmute(arr.get_multiple(k, size_of::<usize>())) };
                    assert_eq!(*r, k + j);
                }
                arr.clear(i - j, 0, size_of::<usize>());
                i = 0;
            }
            arr.remain_settle(0..i, i, rng.gen_range(0..100), size_of::<usize>());
            if rng.gen_range(0..200) == 0 {
                arr.clear(i, 0, size_of::<usize>());
                i = 0;
            }
            check3(&arr, i);
        }
        println!("test3 arr.vec_capacity(): {}", arr.vec_capacity());
    }
    fn check3(arr: &Arr<u8>, len: usize) {
        for i in 0..len {
            let r: &mut usize = unsafe { transmute(arr.get_multiple(i, size_of::<usize>())) };
            assert_eq!(*r, i);
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
        let arr = Arc::new(crate::BucketArr::new());

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
            let arr = crate::BucketArr::new();
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
        let arr = Arc::new(crate::BucketArr::new());

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
        assert_eq!(Location::of(31).bucket, 0);
        assert_eq!(Location::of(31).entry, 31);
        assert_eq!(Location::of(31).len, 32);
        assert_eq!(Location::of(32).bucket, 1);
        assert_eq!(Location::of(32).entry, 0);
        assert_eq!(Location::of(32).len, 64);
        assert_eq!(Location::bucket_len(0), 32);
        assert_eq!(0usize.saturating_sub(0), 0);
        assert_eq!(Location::new(-1, 32, 1).index(0), 1);

        for i in 0..32 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 32);
            assert_eq!(loc.bucket, 0);
            assert_eq!(loc.entry, i);
            assert_eq!(loc.index(0), i);
            assert_eq!(Location::bucket(i), loc.bucket as usize);
            assert_eq!(Location::bucket_capacity(loc.bucket as usize), 32);
        }

        assert_eq!(Location::bucket_len(1), 64);
        for i in 33..96 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 64);
            assert_eq!(loc.bucket, 1);
            assert_eq!(loc.entry, i - 32);
            assert_eq!(loc.index(0), i);
            assert_eq!(Location::bucket(i), loc.bucket as usize);
            assert_eq!(Location::bucket_capacity(loc.bucket as usize), 96);
        }

        assert_eq!(Location::bucket_len(2), 128);
        for i in 96..224 {
            let loc = Location::of(i);
            assert_eq!(loc.len, 128);
            assert_eq!(loc.bucket, 2);
            assert_eq!(loc.entry, i - 96);
            assert_eq!(loc.index(0), i);
            assert_eq!(Location::bucket(i), loc.bucket as usize);
            assert_eq!(Location::bucket_capacity(loc.bucket as usize), 224);
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
        let mut arr = BucketArr::new();
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
