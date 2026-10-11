# pi_arr

默认初始化的自动扩展数组。默认 `Arr<T> = SegmentedArr<T>`，启用 `rc` 后为单线程 `ContiguousArr<T>`。两种具体类型均可直接使用。

## 存储与访问

- `SegmentedArr` 拥有已初始化的连续 boxed slice 分配与 `Buckets<T>`。私有 `OwnedSlice<T> { ptr: NonNull<[T]> }` 通过 `Box::into_raw` 接管唯一所有权，消费或析构时才用 `Box::from_raw` 恢复；字段不直接持有 Box，也不是 Vec。连续区访问比较索引与初始化长度后读取缓存指针，扩展桶不会移动已有元素。
- `with_capacity(n)` 默认初始化恰好 n 个连续元素，`contiguous_len()` 返回初始化长度而非分配器容量。
- 没有逻辑长度，已分配桶中所有默认值均可访问，未分配桶返回 None。Extend 从零开始覆盖。
- `get`、`slice` 和 `segments` 只提供共享引用；`get_mut`、`alloc`、`set` 要求独占数组。
- `slice(range)` 统一遍历连续区与桶并跳过未分配空隙；`segments(range)` 返回裁剪后的连续共享切片。迭代器 `index() - 1` 为刚返回元素的实际索引。
- ZST 遵守正常初始化、Clone、Drop 规则，不再假定无限个默认实例存在。

## 共享访问

`load`、`load_unchecked`、`load_alloc`、`insert` 均为 unsafe。所选元素在返回可变引用存活期间不得有其他引用或并发访问；迭代器和切片也会借用元素。调用方负责同步，整理与释放必须等待所有借用结束。

`SegmentedArr::load_alloc_ptr(index)` 安全返回借用裸指针，必要时并发分配完整默认初始化桶，不创建可变引用。解引用须遵守别名、同步和生命周期规则，不得释放指针。可用于未来 SlotMap 或内部可变元素。

共享引用、slice/segments 与裸前缀迭代允许通过元素内部的 UnsafeCell 合法修改，例如 Atomic 或 Mutex 的共享方法；跨线程仍须正确同步。禁止在重叠共享借用存活期间构造整个元素的 `&mut T` 或替换整个元素，即使 T 含有 UnsafeCell。裸前缀迭代的“无冲突修改”要求不禁止这类合法内部可变访问。

`ContiguousArr::load_alloc_ptr` 为 unsafe。共享增长时整个数组不得有存活引用、迭代器、segments 或将继续使用的旧裸指针，也不得有任何并发访问。增长使所有旧地址失效。ContiguousArr 不实现 Sync，安全共享方法不扩容。

SegmentedArr 的 Send 要求 T: Send，Sync 要求 T: Send + Sync。

## 整理与 Panic

`remain_settle(range, len, additional)` 将保留范围前移到零，未分配空隙补默认值。连续区至少覆盖保留长度加 additional，并保留原连续长度下限。要求 range.start <= range.end <= len。

`settle(len, additional)` 仅保留 0..len：未分配空隙与 len 之后的整个连续尾部均重新默认初始化，两种实现语义一致。`clear(len, additional)` 析构全部旧元素与桶，再重建至少原连续长度及 len + additional 的默认连续区。

Default 或 Clone panic 时 RAII 回收已初始化值，克隆源始终保持所有权。整理与清理不是事务：panic 后数据可能已释放或只剩部分初始化连续区，但数组仍有效，可再次使用和析构。析构 panic 不提供事务回滚；展开期间再次 panic 遵循 Rust 终止规则。

## 0.31.0 API 迁移

此版本直接重命名，不保留旧别名或转发方法。使用 Cargo 默认兼容约束 `pi_arr = "0.30"` 的下游不会自动升级到 0.31；主动升级需要迁移：

| 0.30 名称 | 0.31 名称 |
| --- | --- |
| `VecArr` | `ContiguousArr` |
| `VBArr` | `SegmentedArr` |
| `VecIter` | `ContiguousIter` |
| `vec_capacity()` | `contiguous_len()` |
| `set_vec_capacity(n)` | `resize_contiguous(n)`（仍为 unsafe） |
| `VBArr::capacity(len)` | `SegmentedArr::capacity_for_len(len)` |

移除连续实现中忽略参数的 `capacity(len)`，改用 `contiguous_len()`。`Arr` 与 `Iter` 的 feature 选择保持不变。

非 ZST 的正常增长与整理保留 Vec 扩容产生的完整容量，多余槽位全部默认初始化，转回 Box 时不收缩。`contiguous_len()` 可能大于请求下限。ZST 仍保持有限长度；`with_capacity(n)` 和显式 `resize_contiguous(n)` 保持精确长度。

- 移除公开拥有所有权的 buckets 裸指针，外部不能替换或释放内部桶。
- 共享可变方法改为 unsafe，须先证明完整别名与生命周期契约，不能机械增加 unsafe 块。
- ContiguousIter::Item 从 &mut T 修正为 &T。
- to_vec(ptr, len) 改为 unsafe，要求唯一所有权和准确的原 Vec 布局、容量；内部不再使用它。
- ContiguousArr::resize_contiguous 保留 unsafe 签名，但实际调整初始化长度：增长默认初始化，缩小析构截去元素。
- ZST 的 contiguous_len 返回有限的初始化长度，get 不再返回凭空构造的实例。
- 当前依赖包括 pi_share 0.6、本地 ../pi_vec_remain 0.2 和 ../pi_buckets 0.3.1；pi_share 0.6 使用 nightly 特性。

## 示例

```rust
use pi_arr::{arr, SegmentedArr};
use std::sync::atomic::{AtomicUsize, Ordering};

let mut a = arr![1, 2, 3];
a.set(1, 4);
assert_eq!(a.slice(0..3).copied().collect::<Vec<_>>(), [1, 4, 3]);

let slots = SegmentedArr::<AtomicUsize>::with_capacity(8);
let ptr = slots.load_alloc_ptr(33);
// 通过 AtomicUsize 内部可变性访问，不构造独占引用。
unsafe { &*ptr }.store(7, Ordering::Relaxed);
assert_eq!(slots[33].load(Ordering::Relaxed), 7);
```

## 0.31.1 修复与验证

0.31.0 在取得 Box 切片缓存指针后移动 Box，Stacked Borrows 的 Unique retag 会使缓存 tag 失效。仅将所有者放进 `UnsafeCell<Box<[T]>>` 不改变 pointee 所有权；本次实测 `Box<[UnsafeCell<T>]>` 移动后也仍失败。0.31.1 改用上述显式 boxed-slice owner，移动数组只移动裸所有者，不重新借用元素。借用访问不构造覆盖整个连续区的可变切片；Clone 的共享切片仅存在于克隆调用期间，仍要求没有冲突写入。

保留 Box 切片的长度元数据、紧凑布局和原分配，无每元素大小开销。扩容与整理临时恢复 Vec，仍初始化完整扩容容量后转回 boxed slice；精确初始化、显式 resize、首桶零拷贝接管、ZST 和 panic 回收语义不变。0.31.1 仅完成本地修复，未执行 publish。

在 `nightly-2026-08-17-i686-pc-windows-gnu` 验证默认/rc 的 Debug、Release，以及 Miri Stacked Borrows、Tree Borrows 全量测试：每配置 25 个集成测试和 2 个 compile-fail 文档测试通过。pi_arr 未使用 `-Zmiri-ignore-leaks`。新增回归覆盖数组移动、Vec 重分配、旧裸指针、不相交借用和 Clone 后源指针访问；现有用例覆盖容量、Default/Clone/Drop panic 和 ZST。

当前 stable 构建受 pi_share 0.6 的 nightly 特性阻塞。Miri 的两种借用模型仍属实验模型，测试通过不是所有 unsafe 路径无 UB 的形式证明；未运行单独的 pi_arr 性能基准，不作 pi_arr 性能结论。pi_slot README 另记录使用本地 pi_arr 0.31.1 的预分配槽表微基准，该结果不是 pi_arr 修复前后的独立对照，也不覆盖数组增长、整理或分配/析构成本。
