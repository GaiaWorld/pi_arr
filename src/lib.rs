//! 默认初始化的连续主数组与稳定地址扩展桶。
//! 共享读取可以与分桶分配并存；共享写入必须遵守 unsafe 的别名与同步契约。
//! rc 特性选择单线程 VecArr，共享扩容不得与任何存活引用重叠。
//!
//! 共享可变访问必须显式使用 unsafe：
//! ```compile_fail
//! let arr = pi_arr::Arr::<usize>::with_capacity(2);
//! arr.load(0);
//! ```
//! ```compile_fail
//! let arr = pi_arr::Arr::<usize>::new();
//! arr.insert(0, 1);
//! ```

use pi_buckets::{BucketIter, BucketSegments, Buckets, Location, MAX_ENTRIES, SKIP};
use pi_vec_remain::VecRemain;
use std::cell::{Cell, UnsafeCell};
use std::mem::{replace, take};
use std::ops::{Index, IndexMut, Range};

#[macro_export]
macro_rules! arr {
    () => { $crate::Arr::new() };
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
#[cfg(not(feature = "rc"))]
pub type Arr<T> = VBArr<T>;
#[cfg(feature = "rc")]
pub type Iter<'a, T> = VecIter<'a, T>;
#[cfg(not(feature = "rc"))]
pub type Iter<'a, T> = BucketIter<'a, T>;

/// 已初始化的连续区与稳定地址桶，无逻辑长度；Extend 从零覆盖。
pub struct VBArr<T> {
    vec: Vec<T>,
    ptr: *mut T,
    buckets: Buckets<T>,
}
// 所有权转移允许 T 转移；共享分桶初始化还要求 T 可跨线程转移。
unsafe impl<T: Send> Send for VBArr<T> {}
unsafe impl<T: Send + Sync> Sync for VBArr<T> {}
impl<T> Default for VBArr<T> {
    fn default() -> Self {
        let mut vec = Vec::new();
        let ptr = vec.as_mut_ptr();
        Self {
            vec,
            ptr,
            buckets: Buckets::new(),
        }
    }
}
impl<T> VBArr<T> {
    /// 连续区已初始化的元素数，不是 Vec 分配器的容量（ZST 同样如此）。
    #[inline(always)]
    pub fn vec_capacity(&self) -> usize {
        self.vec.len()
    }
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.vec.len() {
            // 连续区全部初始化；指针只在独占整理时更新。
            return Some(unsafe { &*self.ptr.add(index) });
        }
        let offset = index - self.vec.len();
        if offset > MAX_ENTRIES {
            return None;
        }
        self.buckets.get(&Location::of(offset))
    }
    #[inline(always)]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.vec.len() {
            // self 独占且元素已初始化。
            return Some(unsafe { &mut *self.ptr.add(index) });
        }
        let offset = index - self.vec.len();
        if offset > MAX_ENTRIES {
            return None;
        }
        self.buckets.get_mut(&Location::of(offset))
    }
    /// # Safety
    /// index 必须已分配；引用存活期间不得有冲突写入。
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        if index < self.vec.len() {
            // 调用方保证该元素不存在冲突访问。
            return unsafe { &*self.ptr.add(index) };
        }
        // 调用方保证桶已分配。
        unsafe {
            self.buckets
                .get_unchecked(&Location::of(index - self.vec.len()))
        }
    }
    /// # Safety
    /// index 必须已分配。
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.get_mut(index).expect("element not allocated")
    }
    /// # Safety
    /// 返回引用存活期间该元素不得存在其他引用或并发访问。调用方必须同步
    /// 跨线程访问，并保证数组及分配存活；迭代器和 segments 也会借用元素。
    #[allow(clippy::mut_from_ref)] // unsafe 契约要求元素独占访问。
    #[inline(always)]
    pub unsafe fn load(&self, index: usize) -> Option<&mut T> {
        if index < self.vec.len() {
            // 调用方保证元素独占访问，指针来自原始可变分配。
            return Some(unsafe { &mut *self.ptr.add(index) });
        }
        let offset = index - self.vec.len();
        if offset > MAX_ENTRIES {
            return None;
        }
        // 同样的独占访问契约传递给桶。
        unsafe { self.buckets.load(&Location::of(offset)) }
    }
    /// # Safety
    /// index 必须已分配，并满足 load 的完整生命周期、别名和线程契约。
    #[allow(clippy::mut_from_ref)] // 同 load 的独占元素契约。
    pub unsafe fn load_unchecked(&self, index: usize) -> &mut T {
        // 调用方保证已分配且无冲突访问。
        unsafe { self.load(index) }.expect("element not allocated")
    }
    /// 统一遍历连续区与已分配桶，跳过未分配的空隙。
    /// 可通过元素内部 UnsafeCell 合法写入（跨线程须同步）；共享借用存活期间
    /// 不得构造重叠元素的 &mut T 或替换整个元素，即使 T 含有 UnsafeCell。
    pub fn slice(&self, range: Range<usize>) -> BucketIter<'_, T> {
        // 连续区已初始化且受 self 借用约束；不创建覆盖整个主数组的引用。
        unsafe { BucketIter::from_raw_prefix(self.ptr, self.vec.len(), &self.buckets, range) }
    }
    pub fn segments(&self, range: Range<usize>) -> BucketSegments<'_, T> {
        self.slice(range).into_segments()
    }
}
impl<T: Default> VBArr<T> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = initialized(capacity);
        let ptr = vec.as_mut_ptr();
        Self {
            vec,
            ptr,
            buckets: Buckets::new(),
        }
    }
    /// 给定元素数对应的连续区加完整桶容量；len 是长度而非索引。
    pub fn capacity(&self, len: usize) -> usize {
        if len <= self.vec.len() {
            return self.vec.len();
        }
        self.vec
            .len()
            .checked_add(Location::bucket_capacity(Location::bucket(
                len - self.vec.len() - 1,
            )))
            .expect("capacity overflow")
    }
    #[inline(always)]
    pub fn alloc(&mut self, index: usize) -> &mut T {
        if index < self.vec.len() {
            // self 独占且连续区已初始化。
            return unsafe { &mut *self.ptr.add(index) };
        }
        self.buckets.alloc(&Location::of(index - self.vec.len()))
    }
    /// 返回已初始化元素的借用裸指针；分桶分配可与共享读取并发。
    /// 指针不得释放或超出 self 生命周期，访问必须满足边界、别名与同步规则。
    /// 整理或 clear 会使指针失效，不能与指针的使用重叠。
    #[inline(always)]
    pub fn load_alloc_ptr(&self, index: usize) -> *mut T {
        if index < self.vec.len() {
            // 仅计算已初始化连续区内的指针，不创建引用。
            return unsafe { self.ptr.add(index) };
        }
        let loc = Location::of(index - self.vec.len());
        // 分配器返回完整默认初始化桶，loc.entry 在桶内。
        unsafe { self.buckets.load_alloc_bucket(&loc).add(loc.entry()) }
    }
    /// # Safety
    /// 满足 load 的完整契约：返回引用期间元素必须独占，其他元素可以独立访问。
    #[allow(clippy::mut_from_ref)] // 调用方保证所选元素的独占借用。
    pub unsafe fn load_alloc(&self, index: usize) -> &mut T {
        // 调用方保证无冲突别名，分配已完全初始化。
        unsafe { &mut *self.load_alloc_ptr(index) }
    }
    /// # Safety
    /// 替换期间（包括旧值析构）所选元素不得有其他引用或访问；满足 load 契约。
    pub unsafe fn insert(&self, index: usize, value: T) -> T {
        // 调用方保证所选元素独占。
        unsafe { replace(self.load_alloc(index), value) }
    }
    pub fn set(&mut self, index: usize, value: T) -> T {
        replace(self.alloc(index), value)
    }
    /// 保留范围并前移，空隙补默认值；额外连续区也全部初始化。
    /// Default panic 后 self 仍有效，但原数据可能已释放。
    pub fn remain_settle(&mut self, range: Range<usize>, len: usize, additional: usize) {
        assert!(
            range.start <= range.end && range.end <= len,
            "invalid retained range"
        );
        let target = range
            .len()
            .checked_add(additional)
            .expect("capacity overflow")
            .max(self.vec.len());
        let prefix_len = self.vec.len();
        // 先解除 self 的所有权，再调用可能 panic 的 Default/Drop；局部 RAII 负责回收。
        let mut result = take(&mut self.vec);
        self.ptr = self.vec.as_mut_ptr();
        let buckets = self.buckets.take();
        result.reserve(target - result.len());
        result.remain(range.clone());
        for (bucket, values) in buckets.into_iter().enumerate() {
            if values.is_empty() {
                continue;
            }
            let start = prefix_len
                .checked_add(Location::bucket_len(bucket) - SKIP)
                .expect("index overflow");
            append_retained(values, start, &range, &mut result);
        }
        result.resize_with(target, T::default);
        self.ptr = result.as_mut_ptr();
        self.vec = result;
    }
    pub fn settle(&mut self, len: usize, additional: usize) {
        self.remain_settle(0..len, len, additional);
    }
    /// 释放所有旧值与桶，保留至少原连续长度及 len + additional 个默认元素。
    /// Default panic 后 self 为空且仍可使用。
    pub fn clear(&mut self, len: usize, additional: usize) {
        let target = len
            .checked_add(additional)
            .expect("capacity overflow")
            .max(self.vec.len());
        let prefix = take(&mut self.vec);
        self.ptr = self.vec.as_mut_ptr();
        let buckets = self.buckets.take();
        drop(prefix);
        drop(buckets);
        let mut vec = initialized(target);
        self.ptr = vec.as_mut_ptr();
        self.vec = vec;
    }
}
impl<T: Clone> Clone for VBArr<T> {
    fn clone(&self) -> Self {
        let mut vec = self.vec.clone();
        let ptr = vec.as_mut_ptr();
        // 源存储始终由 self 持有，克隆失败由局部 Vec 回收。
        Self {
            vec,
            ptr,
            buckets: self.buckets.clone(),
        }
    }
}

fn initialized<T: Default>(len: usize) -> Vec<T> {
    let mut vec = Vec::with_capacity(len);
    vec.resize_with(len, T::default);
    vec
}
fn append_retained<T: Default>(
    mut values: Vec<T>,
    start: usize,
    range: &Range<usize>,
    result: &mut Vec<T>,
) {
    let left = range.start.saturating_sub(start).min(values.len());
    let right = range.end.saturating_sub(start).min(values.len());
    if left >= right {
        return;
    }
    let offset = start.saturating_sub(range.start);
    result.resize_with(offset.max(result.len()), T::default);
    values.remain_to(left..right, result);
}

/// 单线程连续数组。共享扩容仅通过 unsafe 接口，安全接口不会移动共享存储。
pub struct VecArr<T> {
    vec: UnsafeCell<Vec<T>>,
    ptr: Cell<*mut T>,
    len: Cell<usize>,
}
// VecArr 不实现 Sync；所有权转移只要求 T: Send。
unsafe impl<T: Send> Send for VecArr<T> {}
impl<T> Default for VecArr<T> {
    fn default() -> Self {
        let mut vec = Vec::new();
        let ptr = vec.as_mut_ptr();
        Self {
            vec: UnsafeCell::new(vec),
            ptr: Cell::new(ptr),
            len: Cell::new(0),
        }
    }
}
impl<T> VecArr<T> {
    pub fn vec_capacity(&self) -> usize {
        self.len.get()
    }
    pub fn capacity(&self, _len: usize) -> usize {
        self.len.get()
    }
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len.get() {
            return None;
        }
        // 缓存长度仅覆盖已初始化元素，安全共享方法不会扩容。
        Some(unsafe { &*self.ptr.get().add(index) })
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len.get() {
            return None;
        }
        // self 独占且 index 已验证。
        Some(unsafe { &mut *self.ptr.get().add(index) })
    }
    /// # Safety
    /// index 必须小于 vec_capacity；引用存活期间不得扩容、释放或冲突写入。
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // 调用方保证已初始化的有效索引与无冲突访问。
        unsafe { &*self.ptr.get().add(index) }
    }
    /// # Safety
    /// index 必须小于 vec_capacity。
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // 独占 self，调用方保证有效索引。
        unsafe { &mut *self.ptr.get().add(index) }
    }
    /// # Safety
    /// 返回引用期间元素必须独占访问，数组不得增长、整理、释放。
    /// VecArr 不是 Sync；不得并发访问任何共享扩容接口。
    #[allow(clippy::mut_from_ref)] // unsafe 要求所选元素独占访问。
    pub unsafe fn load(&self, index: usize) -> Option<&mut T> {
        if index >= self.len.get() {
            return None;
        }
        // 调用方保证元素无其他别名。
        Some(unsafe { &mut *self.ptr.get().add(index) })
    }
    /// # Safety
    /// index 已初始化且满足 load 完整契约。
    #[allow(clippy::mut_from_ref)] // 同 load 契约。
    pub unsafe fn load_unchecked(&self, index: usize) -> &mut T {
        // 调用方保证元素已分配且独占。
        unsafe { self.load(index) }.expect("element not allocated")
    }
    pub fn slice(&self, range: Range<usize>) -> VecIter<'_, T> {
        assert!(range.start <= range.end, "reversed range");
        let end = range.end.min(self.len.get());
        let start = range.start.min(end);
        // 缓存指针覆盖已初始化区间，迭代器生命周期绑定 self。
        let slice = unsafe { std::slice::from_raw_parts(self.ptr.get().add(start), end - start) };
        VecIter {
            iter: slice.iter(),
            index: start,
        }
    }
    pub fn segments(&self, range: Range<usize>) -> impl Iterator<Item = &[T]> {
        let iter = self.slice(range);
        let slice = iter.iter.as_slice();
        (!slice.is_empty()).then_some(slice).into_iter()
    }
}
impl<T: Default> VecArr<T> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = initialized(capacity);
        let ptr = vec.as_mut_ptr();
        Self {
            vec: UnsafeCell::new(vec),
            ptr: Cell::new(ptr),
            len: Cell::new(capacity),
        }
    }
    /// 改变已初始化长度，不伪造分配器容量。
    /// # Safety
    /// 保留此 unsafe 签名；self 独占，缩小会析构截去的元素，增长会默认初始化。
    pub unsafe fn set_vec_capacity(&mut self, capacity: usize) {
        self.resize(capacity);
    }
    fn resize(&mut self, len: usize) {
        let vec = self.vec.get_mut();
        // resize 的 Default 可 panic；先使缓存不再暴露可能失效的旧指针。
        self.len.set(0);
        self.ptr.set(std::ptr::NonNull::dangling().as_ptr());
        // 局部 guard 在 panic 时也更新缓存，保留 Vec 已成功初始化的部分。
        struct Refresh<'a, T> {
            vec: &'a mut Vec<T>,
            ptr: &'a Cell<*mut T>,
            len: &'a Cell<usize>,
        }
        impl<T> Drop for Refresh<'_, T> {
            fn drop(&mut self) {
                self.ptr.set(self.vec.as_mut_ptr());
                self.len.set(self.vec.len());
            }
        }
        let guard = Refresh {
            vec,
            ptr: &self.ptr,
            len: &self.len,
        };
        guard.vec.resize_with(len, T::default);
    }
    pub fn alloc(&mut self, index: usize) -> &mut T {
        if index >= self.len.get() {
            self.resize(index.checked_add(1).expect("index overflow"));
        }
        // self 独占且扩容后元素已初始化。
        unsafe { &mut *self.ptr.get().add(index) }
    }
    pub fn set(&mut self, index: usize, value: T) -> T {
        replace(self.alloc(index), value)
    }
    /// # Safety
    /// 若 index 超出当前连续区，整个数组不得有任何存活引用、迭代器、segments
    /// 或将继续使用的旧裸指针；不得有任何并发数组访问，调用期间独占整个存储。
    /// 不增长时仅返回稳定借用指针，但解引用仍要求有效生命周期、别名和同步。
    /// 指针不得释放或超出 self 生命周期，之后增长会使全部旧指针失效。
    pub unsafe fn load_alloc_ptr(&self, index: usize) -> *mut T {
        if index >= self.len.get() {
            // 调用方保证整个存储独占；先移出 Vec，使 panic 时 self 仍有效。
            let mut vec = unsafe { take(&mut *self.vec.get()) };
            self.len.set(0);
            self.ptr.set(std::ptr::NonNull::dangling().as_ptr());
            vec.resize_with(index.checked_add(1).expect("index overflow"), T::default);
            self.ptr.set(vec.as_mut_ptr());
            self.len.set(vec.len());
            // 独占存储契约允许安装新的所有者。
            unsafe {
                *self.vec.get() = vec;
            }
        }
        // 分配已初始化，index 在范围内。
        unsafe { self.ptr.get().add(index) }
    }
    /// # Safety
    /// 满足 load_alloc_ptr 完整契约，并在返回引用存活期间独占该元素；
    /// 任意元素引用存活期间都禁止共享增长。
    #[allow(clippy::mut_from_ref)] // 调用方同时保证增长独占与元素独占。
    pub unsafe fn load_alloc(&self, index: usize) -> &mut T {
        // 调用方保证独占所需存储与元素。
        unsafe { &mut *self.load_alloc_ptr(index) }
    }
    /// # Safety
    /// 满足 load_alloc 的全部契约，包括增长时不存在任何数组引用。
    pub unsafe fn insert(&self, index: usize, value: T) -> T {
        // 调用方保证增长与替换没有冲突访问。
        unsafe { replace(self.load_alloc(index), value) }
    }
    pub fn remain_settle(&mut self, range: Range<usize>, len: usize, additional: usize) {
        assert!(
            range.start <= range.end && range.end <= len,
            "invalid retained range"
        );
        let target = range
            .len()
            .checked_add(additional)
            .expect("capacity overflow")
            .max(self.len.get());
        let mut result = take(self.vec.get_mut());
        self.len.set(0);
        self.ptr.set(std::ptr::NonNull::dangling().as_ptr());
        result.reserve(target - result.len());
        result.remain(range);
        result.resize_with(target, T::default);
        self.ptr.set(result.as_mut_ptr());
        self.len.set(result.len());
        *self.vec.get_mut() = result;
    }
    pub fn settle(&mut self, len: usize, additional: usize) {
        self.remain_settle(0..len, len, additional);
    }
    pub fn clear(&mut self, len: usize, additional: usize) {
        let target = len
            .checked_add(additional)
            .expect("capacity overflow")
            .max(self.len.get());
        let old = take(self.vec.get_mut());
        self.len.set(0);
        self.ptr.set(std::ptr::NonNull::dangling().as_ptr());
        drop(old);
        self.resize(target);
    }
}
impl<T: Clone> Clone for VecArr<T> {
    fn clone(&self) -> Self {
        // 安全共享方法不改动 Vec；unsafe 增长契约排除当前共享借用。
        let mut vec = unsafe { &*self.vec.get() }.clone();
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        Self {
            vec: UnsafeCell::new(vec),
            ptr: Cell::new(ptr),
            len: Cell::new(len),
        }
    }
}

pub struct VecIter<'a, T> {
    iter: std::slice::Iter<'a, T>,
    index: usize,
}
impl<T> VecIter<'_, T> {
    pub fn empty() -> Self {
        Self {
            iter: [].iter(),
            index: 0,
        }
    }
    pub fn index(&self) -> usize {
        self.index
    }
}
impl<'a, T> Iterator for VecIter<'a, T> {
    type Item = &'a T;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()?;
        self.index += 1;
        Some(value)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
impl<T> ExactSizeIterator for VecIter<'_, T> {}

macro_rules! impl_array_traits {
    ($name:ident) => {
        impl<T> Index<usize> for $name<T> {
            type Output = T;
            fn index(&self, index: usize) -> &T {
                self.get(index).expect("element not allocated")
            }
        }
        impl<T> IndexMut<usize> for $name<T> {
            fn index_mut(&mut self, index: usize) -> &mut T {
                self.get_mut(index).expect("element not allocated")
            }
        }
        impl<T: Default> FromIterator<T> for $name<T> {
            fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
                let iter = iter.into_iter();
                let mut result = Self::with_capacity(iter.size_hint().0);
                result.extend(iter);
                result
            }
        }
        impl<T: Default> Extend<T> for $name<T> {
            fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
                for (i, value) in iter.into_iter().enumerate() {
                    self.set(i, value);
                }
            }
        }
    };
}
impl_array_traits!(VBArr);
impl_array_traits!(VecArr);

/// 从裸指针接管 Vec 所有权；内部实现不使用此方法。
/// # Safety
/// ptr 必须是以 Vec<T> 的布局与分配器分配的唯一所有者，分配容量恰为 len，
/// 前 len 个元素已初始化。满足 Vec::from_raw_parts 的全部前提（包括 ZST），
/// 不得仍有引用、另一所有者或再次释放该分配。即使 len 为零也要求非空对齐指针。
pub unsafe fn to_vec<T>(ptr: *mut T, len: usize) -> Vec<T> {
    // 所有原始分配、布局与唯一所有权前提由调用方保证。
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
