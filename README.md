# pi_arr

默认初始化的自动扩展数组。默认 `Arr<T> = VBArr<T>`，启用 `rc` 后为单线程 `VecArr<T>`。两种具体类型均可直接使用。

## 存储与访问

- `VBArr` 拥有已初始化的连续 `Vec<T>` 与 `Buckets<T>`。连续区访问比较索引与初始化长度后读取缓存指针，扩展桶不会移动已有元素。
- `with_capacity(n)` 默认初始化恰好 n 个连续元素，`vec_capacity()` 返回初始化长度而非分配器容量。
- 没有逻辑长度，已分配桶中所有默认值均可访问，未分配桶返回 None。Extend 从零开始覆盖。
- `get`、`slice` 和 `segments` 只提供共享引用；`get_mut`、`alloc`、`set` 要求独占数组。
- `slice(range)` 统一遍历连续区与桶并跳过未分配空隙；`segments(range)` 返回裁剪后的连续共享切片。迭代器 `index() - 1` 为刚返回元素的实际索引。
- ZST 遵守正常初始化、Clone、Drop 规则，不再假定无限个默认实例存在。

## 共享访问

`load`、`load_unchecked`、`load_alloc`、`insert` 均为 unsafe。所选元素在返回可变引用存活期间不得有其他引用或并发访问；迭代器和切片也会借用元素。调用方负责同步，整理与释放必须等待所有借用结束。

`VBArr::load_alloc_ptr(index)` 安全返回借用裸指针，必要时并发分配完整默认初始化桶，不创建可变引用。解引用须遵守别名、同步和生命周期规则，不得释放指针。可用于未来 SlotMap 或内部可变元素。

共享引用、slice/segments 与裸前缀迭代允许通过元素内部的 UnsafeCell 合法修改，例如 Atomic 或 Mutex 的共享方法；跨线程仍须正确同步。禁止在重叠共享借用存活期间构造整个元素的 `&mut T` 或替换整个元素，即使 T 含有 UnsafeCell。裸前缀迭代的“无冲突修改”要求不禁止这类合法内部可变访问。

`VecArr::load_alloc_ptr` 为 unsafe。共享增长时整个数组不得有存活引用、迭代器、segments 或将继续使用的旧裸指针，也不得有任何并发访问。增长使所有旧地址失效。VecArr 不实现 Sync，安全共享方法不扩容。

VBArr 的 Send 要求 T: Send，Sync 要求 T: Send + Sync。

## 整理与 Panic

`remain_settle(range, len, additional)` 将保留范围前移到零，未分配空隙补默认值。连续区至少覆盖保留长度加 additional，并保留原连续长度下限。要求 range.start <= range.end <= len。

`settle(len, additional)` 仅保留 0..len：未分配空隙与 len 之后的整个连续尾部均重新默认初始化，两种实现语义一致。`clear(len, additional)` 析构全部旧元素与桶，再重建至少原连续长度及 len + additional 的默认连续区。

Default 或 Clone panic 时 RAII 回收已初始化值，克隆源始终保持所有权。整理与清理不是事务：panic 后数据可能已释放或只剩部分初始化连续区，但数组仍有效，可再次使用和析构。析构 panic 不提供事务回滚；展开期间再次 panic 遵循 Rust 终止规则。

## API 迁移

- 移除公开拥有所有权的 buckets 裸指针，外部不能替换或释放内部桶。
- 共享可变方法改为 unsafe，须先证明完整别名与生命周期契约，不能机械增加 unsafe 块。
- VecIter::Item 从 &mut T 修正为 &T。
- to_vec(ptr, len) 改为 unsafe，要求唯一所有权和准确的原 Vec 布局、容量；内部不再使用它。
- VecArr::set_vec_capacity 保留 unsafe 签名，但实际调整初始化长度：增长默认初始化，缩小析构截去元素。
- ZST 的 vec_capacity 返回有限的初始化长度，get 不再返回凭空构造的实例。
- 移除 nightly 特性和未使用的 pi_share、pi_vec_remain；依赖本地 ../pi_buckets 修复版本，无新增依赖。

## 示例

```rust
use pi_arr::{arr, VBArr};
use std::sync::atomic::{AtomicUsize, Ordering};

let mut a = arr![1, 2, 3];
a.set(1, 4);
assert_eq!(a.slice(0..3).copied().collect::<Vec<_>>(), [1, 4, 3]);

let slots = VBArr::<AtomicUsize>::with_capacity(8);
let ptr = slots.load_alloc_ptr(33);
// 通过 AtomicUsize 内部可变性访问，不构造独占引用。
unsafe { &*ptr }.store(7, Ordering::Relaxed);
assert_eq!(slots[33].load(Ordering::Relaxed), 7);
```

测试均位于 tests/，支持 stable i686 默认与 rc 配置及 Miri Tree Borrows。此阶段未运行性能基准，不宣称未经测量的纳秒指标。
